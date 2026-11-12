{-# LANGUAGE NoImplicitPrelude #-}

-- | Oracles for the circuits-ecosystem rewrite of circuits-net.
--
--   1. Identity-training oracle: a 2-layer MLP trained on the identity
--      mapping must decrease loss.
--   2. Packaged-training oracle: 'trainNetCoalgebra', the delegate of the
--      same training loop, must also decrease loss.
--   3. Gradient oracle: gradients produced by 'circuits-ad' via 'DiffP'
--      must agree with hand-rolled reference gradients on a fixed input.
--   4. Nguyen Lemma 3.3 oracle: a fixed two-layer reconstruction must
--      reproduce an independently chosen target through leaky ReLU.
module Main where

import Circuit.Mat.Dense (Matrix (..), fromLists, matPlus, matTimes, toLists)
import Data.List qualified as List
import Harpie.Array (arrayAs)
import Harpie.Array qualified as A
import Net (NetParams (..), netParamsFromArrays)
import NetCoalgebra
  ( forward,
    gradientsViaAD,
    mseLoss,
    referenceGradients,
    trainIdentity,
    trainNetCoalgebra,
  )
import System.Exit (exitFailure, exitSuccess)
import Prelude hiding (id, (.))

main :: IO ()
main = do
  -- Create small net: 4 → 8 → 4 (identity mapping)
  let w1Init = A.tabulate [8, 4] (const 1.0) :: A.Array Double
  let b1Init = A.tabulate [8] (const 0.0) :: A.Array Double
  let w2Init = A.tabulate [4, 8] (const 1.0) :: A.Array Double
  let b2Init = A.tabulate [4] (const 0.0) :: A.Array Double
  let params = netParamsFromArrays w1Init b1Init w2Init b2Init

  -- Input/target: all-ones vector
  let x = A.tabulate [4] (const 1.0) :: A.Array Double
  let target = A.tabulate [4] (const 1.0) :: A.Array Double

  -- Train for 500 steps
  let lr = 0.01
  let history = trainIdentity 500 lr params x target
  let losses = map fst history
      initialLoss = case losses of (l : _) -> l; [] -> 0
      finalLoss = case reverse losses of (l : _) -> l; [] -> 0

  putStrLn $ "Initial loss: " ++ show initialLoss
  putStrLn $ "Final loss:   " ++ show finalLoss
  putStrLn $ "Steps:        " ++ show (length history)

  -- Oracle 1: net loss decrease
  if finalLoss < initialLoss / 2
    then putStrLn "PASS: loss decreased substantially"
    else do
      putStrLn $ "FAIL: loss " ++ show initialLoss ++ " -> " ++ show finalLoss
      exitFailure

  -- Oracle 2: final loss is small
  if finalLoss < 0.1
    then putStrLn "PASS: final loss < 0.1"
    else do
      putStrLn $ "FAIL: final loss " ++ show finalLoss ++ " >= 0.1"
      exitFailure

  -- Oracle 3: forward pass produces correct shape
  let y = forward params x
  if A.size y == 4
    then putStrLn "PASS: output shape correct (4)"
    else do
      putStrLn $ "FAIL: output shape " ++ show (A.size y) ++ " /= 4"
      exitFailure

  ----------------------------------------------------------------------
  -- Packaged training oracle
  ----------------------------------------------------------------------
  do
    let packagedHistory = trainNetCoalgebra params x target lr 500
        packagedInitial = fst (mseLoss (forward params x) target)
        packagedFinal = case packagedHistory of
          [] -> packagedInitial
          hs -> fst (mseLoss (forward (snd (last hs)) x) target)
    putStrLn $ "Packaged  initial loss: " ++ show packagedInitial
    putStrLn $ "Packaged  final loss:   " ++ show packagedFinal

    if packagedFinal < packagedInitial / 2
      then putStrLn "PASS: packaged loss decreased substantially"
      else do
        putStrLn $ "FAIL: packaged loss " ++ show packagedInitial ++ " -> " ++ show packagedFinal
        exitFailure

    if packagedFinal < 0.1
      then putStrLn "PASS: packaged final loss < 0.1"
      else do
        putStrLn $ "FAIL: packaged final loss " ++ show packagedFinal ++ " >= 0.1"
        exitFailure

  ----------------------------------------------------------------------
  -- Gradient oracle: circuits-ad vs hand-rolled reference
  ----------------------------------------------------------------------
  do
    let adGrad = gradientsViaAD params x target
        refGrad = referenceGradients params x target
        adFlat = flatten adGrad
        refFlat = flatten refGrad
        diffs = zipWith (-) adFlat refFlat
        relErrs = zipWith (\a r -> abs (a - r) / max 1e-12 (abs r)) adFlat refFlat
        maxRelErr = maximum relErrs
    putStrLn $ "Gradient vector length: " ++ show (length adFlat)
    putStrLn $ "Max absolute gradient diff: " ++ show (maximum (map abs diffs))
    putStrLn $ "Max relative gradient diff: " ++ show maxRelErr

    if maxRelErr < 1e-10
      then putStrLn "PASS: circuits-ad gradients match reference"
      else do
        putStrLn $ "FAIL: circuits-ad gradients diverge from reference (max rel err " ++ show maxRelErr ++ ")"
        exitFailure

  ----------------------------------------------------------------------
  -- Nguyen Lemma 3.3: fixed L = k = 2 value reconstruction
  ----------------------------------------------------------------------
  nguyenLemma33

  putStrLn "All oracles passed."
  exitSuccess

-- | Flatten a 'NetParams' bundle into a single list for comparison.
flatten :: NetParams Double -> [Double]
flatten p =
  concat
    [ arrayAs (unMatrix (w1 p)),
      arrayAs (b1 p),
      arrayAs (unMatrix (w2 p)),
      arrayAs (b2 p)
    ]

-- | Nguyen's Lemma 3.3 at N = d = 2, n1 = 3, n2 = 2. The target is chosen
-- independently; mixed signs in B1 exercise both leaky-ReLU branches.
nguyenLemma33 :: IO ()
nguyenLemma33 = do
  let x = fromLists [[1, 0], [0, 1]]
      z = fromLists [[1, 0, 1], [0, 1, 1]]
      w2 = fromLists [[1, 0], [0, 1], [1, 1]]
      b2 = fromLists [[0.2, -0.1]]
      targetA = fromLists [[0.7, -0.4], [1.2, 0.9]]
      ones = fromLists [[1], [1]]
      identity2 = fromLists [[1, 0], [0, 1]]
      w2Gram = matTimes (transposeDense w2) w2
      zGram = matTimes z (transposeDense z)
      w2Pinv = matTimes (inverse2 w2Gram) (transposeDense w2)
      zPinv = matTimes (transposeDense z) (inverse2 zGram)
      b1Target = matTimes (matMinus targetA (matTimes ones b2)) w2Pinv
      preB1 = mapDense leakyInverse b1Target
      u = matTimes zPinv preB1
      w1 = takeRows 2 u
      b1 = lastRow u
      f1 = mapDense leaky (matPlus (matTimes x w1) (matTimes ones b1))
      f2 = matPlus (matTimes f1 w2) (matTimes ones b2)
      w2InverseError = maxAbsDiff (matTimes w2Pinv w2) identity2
      zInverseError = maxAbsDiff (matTimes z zPinv) identity2
      inputError = maxAbsDiff (matTimes z u) preB1
      reconstructionError = maxAbsDiff f2 targetA
      tolerance = 1e-12
      b1Values = concat (toLists b1Target)
  putStrLn $ "Nguyen W2 right-inverse max abs diff: " ++ show w2InverseError
  putStrLn $ "Nguyen [X,1] right-inverse max abs diff: " ++ show zInverseError
  putStrLn $ "Nguyen first-layer reconstruction max abs diff: " ++ show inputError
  putStrLn $ "Nguyen F2 reconstruction max abs diff: " ++ show reconstructionError
  requireWitness "Nguyen rank(X) = 2" (abs (det2 x) > tolerance)
  requireWitness "Nguyen W2 has full column rank" (abs (det2 w2Gram) > tolerance)
  requireWitness
    "Nguyen B1 exercises positive and negative leaky-ReLU branches"
    (any (< 0) b1Values && any (> 0) b1Values)
  requireWitness "Nguyen W2 pseudoinverse is a left inverse" (w2InverseError < tolerance)
  requireWitness "Nguyen [X,1] pseudoinverse is a right inverse" (zInverseError < tolerance)
  requireWitness "Nguyen [X,1] U reconstructs sigma^-1(B1)" (inputError < tolerance)
  requireWitness "Nguyen Lemma 3.3 reconstructs the independent target A" (reconstructionError < tolerance)

requireWitness :: String -> Bool -> IO ()
requireWitness label ok =
  if ok
    then putStrLn $ "PASS: " ++ label
    else do
      putStrLn $ "FAIL: " ++ label
      exitFailure

transposeDense :: Matrix a -> Matrix a
transposeDense m = fromLists (List.transpose (toLists m))

mapDense :: (a -> b) -> Matrix a -> Matrix b
mapDense f m = fromLists (map (map f) (toLists m))

matMinus :: (Num a) => Matrix a -> Matrix a -> Matrix a
matMinus a b = fromLists (zipWith (zipWith (-)) (toLists a) (toLists b))

takeRows :: Int -> Matrix a -> Matrix a
takeRows n m = fromLists (take n (toLists m))

lastRow :: Matrix a -> Matrix a
lastRow m = fromLists [last (toLists m)]

maxAbsDiff :: Matrix Double -> Matrix Double -> Double
maxAbsDiff a b = maximum (map abs (concat (toLists (matMinus a b))))

det2 :: Matrix Double -> Double
det2 m = case toLists m of
  [[a, b], [c, d]] -> a * d - b * c
  _ -> error "det2: expected a 2x2 matrix"

-- | Inverse of the fixed witness's 2x2 Gram matrix.
inverse2 :: Matrix Double -> Matrix Double
inverse2 m = case toLists m of
  [[a, b], [c, d]] ->
    let det = det2 m
     in fromLists [[d / det, negate b / det], [negate c / det, a / det]]
  _ -> error "inverse2: expected a 2x2 matrix"

leaky :: Double -> Double
leaky x = if x < 0 then 0.1 * x else x

leakyInverse :: Double -> Double
leakyInverse y = if y < 0 then y / 0.1 else y
