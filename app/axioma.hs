{-# LANGUAGE NoImplicitPrelude #-}

-- | Oracles for the circuits-ecosystem rewrite of circuits-net.
--
--   1. Identity-training smoke: retain the all-ones loss decrease and add
--      deterministic nonconstant identity examples.
--   2. Packaged-training oracle: 'trainNetCoalgebra', the delegate of the
--      same training loop, must also decrease loss.
--   3. Gradient oracles: compare the hand-written paths, then check both ReLU
--      branches against finite differences of an independent direct objective.
--   4. Exact contracts: MSE, public flatten/reshape round trip, one SGD step,
--      Para/DiffP primal parity, and pinned boundary behaviour.
--   5. Dense/mathematical witness: Nguyen Lemma 3.3 fixed reconstruction.
module Main where

import Circuit.Mat.Dense (Matrix (..), fromLists, matPlus, matTimes, toLists)
import Data.List qualified as List
import Harpie.Array (arrayAs)
import Harpie.Array qualified as A
import Net (NetParams (..), netParamsFromArrays)
import NetCoalgebra
  ( forward,
    gradientsViaAD,
    initNetState,
    mseLoss,
    referenceGradients,
    trainIdentity,
    trainNetCoalgebra,
    trainStep,
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
  -- Gradient oracle: manual DiffP vs hand-rolled reference
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
      then putStrLn "PASS: manual DiffP gradients match reference"
      else do
        putStrLn $ "FAIL: manual DiffP gradients diverge from reference (max rel err " ++ show maxRelErr ++ ")"
        exitFailure

  nonconstantIdentityOracles params
  finiteDifferenceOracles
  numericalContractOracles

  ----------------------------------------------------------------------
  -- Dense/mathematical witness: Nguyen Lemma 3.3, fixed L = k = 2.
  -- Kept here as a Dense client witness; no network coverage is claimed.
  ----------------------------------------------------------------------
  putStrLn "Dense/mathematical witness: Nguyen Lemma 3.3"
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

vec :: [Double] -> A.Array Double
vec xs = A.array [length xs] xs

-- | Four fixed nonconstant inputs, each trained against itself.
nonconstantIdentityOracles :: NetParams Double -> IO ()
nonconstantIdentityOracles params = mapM_ checkCase cases
  where
    cases =
      [ ("e0", vec [1, 0, 0, 0]),
        ("e1", vec [0, 1, 0, 0]),
        ("e2", vec [0, 0, 1, 0]),
        ("e3", vec [0, 0, 0, 1])
      ]
    checkCase (label, x) = do
      let history = trainIdentity 500 0.01 params x x
      case history of
        [] -> requireWitness (label ++ " nonconstant identity produced 500 steps") False
        firstEntry : _ -> do
          let initialLoss = fst firstEntry
              finalLoss = List.foldl' (\_ (stepLoss, _) -> stepLoss) initialLoss history
          putStrLn $ label ++ " identity initial loss: " ++ show initialLoss
          putStrLn $ label ++ " identity final loss:   " ++ show finalLoss
          requireWitness
            (label ++ " nonconstant identity loss decreased substantially")
            (length history == 500 && finalLoss < initialLoss / 2)
          requireWitness (label ++ " nonconstant identity final loss < 0.1") (finalLoss < 0.1)

-- | A 1 -> 2 -> 1 parameter bundle in w1, b1, w2, b2 order.
netFromFlat7 :: [Double] -> NetParams Double
netFromFlat7 [w1a, w1b, b1a, b1b, w2a, w2b, b2] =
  netParamsFromArrays
    (A.array [2, 1] [w1a, w1b])
    (A.array [2] [b1a, b1b])
    (A.array [1, 2] [w2a, w2b])
    (A.array [1] [b2])
netFromFlat7 _ = error "netFromFlat7: expected seven parameters"

positiveFlat :: [Double]
positiveFlat = [0.5, 0.25, 0.125, -0.125, 0.75, -0.5, 0.25]

negativeFlat :: [Double]
negativeFlat = [-0.5, -0.25, -0.125, -0.25, 0.75, -0.5, 0.25]

-- | Direct scalar preactivations for the independent reference objective.
preactivations7 :: [Double] -> Double -> (Double, Double)
preactivations7 [w1a, w1b, b1a, b1b, _, _, _] x =
  (w1a * x + b1a, w1b * x + b1b)
preactivations7 _ _ = error "preactivations7: expected seven parameters"

-- | Independent objective: direct scalar arithmetic, with no call to the
-- library's forward, loss, gradient, matrix, or parameter-flattening code.
directLoss7 :: [Double] -> Double -> Double -> Double
directLoss7 flat@[_, _, _, _, w2a, w2b, b2] x target =
  let (zA, zB) = preactivations7 flat x
      hA = max zA 0
      hB = max zB 0
      y = w2a * hA + w2b * hB + b2
   in (y - target) * (y - target)
directLoss7 _ _ _ = error "directLoss7: expected seven parameters"

-- | Central differences of 'directLoss7' at each of the seven parameters.
centralGradient7 :: [Double] -> Double -> Double -> [Double]
centralGradient7 params x target =
  [ (directLoss7 (perturb i h) x target - directLoss7 (perturb i (negate h)) x target) / (2 * h)
  | i <- [0 .. length params - 1]
  ]
  where
    h = 1e-5
    perturb i delta =
      zipWith (\j value -> if i == j then value + delta else value) [0 ..] params

maxAbsListDiff :: [Double] -> [Double] -> Double
maxAbsListDiff xs ys = maximum (zipWith (\x y -> abs (x - y)) xs ys)

finiteDifferenceOracles :: IO ()
finiteDifferenceOracles = do
  let x = 2
      target = 0.5
      (positiveZA, positiveZB) = preactivations7 positiveFlat x
      (negativeZA, negativeZB) = preactivations7 negativeFlat x
      positiveAD = flatten (gradientsViaAD (netFromFlat7 positiveFlat) (vec [x]) (vec [target]))
      negativeAD = flatten (gradientsViaAD (netFromFlat7 negativeFlat) (vec [x]) (vec [target]))
      positiveReference = centralGradient7 positiveFlat x target
      negativeReference = centralGradient7 negativeFlat x target
      positiveError = maxAbsListDiff positiveAD positiveReference
      negativeError = maxAbsListDiff negativeAD negativeReference
      tolerance = 1e-7
  requireWitness
    "positive finite-difference fixture has two positive ReLU preactivations"
    (positiveZA > 0 && positiveZB > 0)
  requireWitness
    "negative finite-difference fixture has two negative ReLU preactivations"
    (negativeZA < 0 && negativeZB < 0)
  putStrLn $ "Positive ReLU finite-difference max abs diff: " ++ show positiveError
  putStrLn $ "Negative ReLU finite-difference max abs diff: " ++ show negativeError
  requireWitness
    "DiffP gradients match independent positive-ReLU finite differences"
    (length positiveAD == length positiveReference && positiveError < tolerance)
  requireWitness
    "DiffP gradients match independent negative-ReLU finite differences"
    (length negativeAD == length negativeReference && negativeError < tolerance)

numericalContractOracles :: IO ()
numericalContractOracles = do
  let (loss, gradient) = mseLoss (vec [1, 3]) (vec [0, 1])
  requireWitness "exact MSE is 5/2" (loss == 2.5)
  requireWitness "exact MSE gradient is [1,2]" (arrayAs gradient == [1, 2])

  let positiveParams = netFromFlat7 positiveFlat
      positiveX = vec [2]
      positiveTarget = vec [0.5]
      roundTrip = trainIdentity 1 0 positiveParams positiveX positiveTarget
  requireWitness
    "zero-rate public training step is an exact flatten/reshape round trip"
    (map snd roundTrip == [positiveParams])

  -- x = w1 = w2 = 1, both biases and target zero: loss = 1 and every
  -- parameter gradient = 2, so one SGD step at rate 1/4 has exact halves.
  let oneStepParams =
        netParamsFromArrays
          (A.array [1, 1] [1])
          (A.array [1] [0])
          (A.array [1, 1] [1])
          (A.array [1] [0])
      expectedParams =
        netParamsFromArrays
          (A.array [1, 1] [0.5])
          (A.array [1] [-0.5])
          (A.array [1, 1] [0.5])
          (A.array [1] [-0.5])
      oneStep = trainIdentity 1 0.25 oneStepParams (vec [1]) (vec [0])
  requireWitness "exact one-step SGD loss is 1" (map fst oneStep == [1])
  requireWitness
    "exact one-step SGD parameters are [w1=1/2,b1=-1/2,w2=1/2,b2=-1/2]"
    (map snd oneStep == [expectedParams])

  let diffPrediction = fst (trainStep (initNetState positiveParams) (positiveX, positiveTarget))
  requireWitness
    "Para and DiffP forward paths agree exactly"
    (forward positiveParams positiveX == diffPrediction)

  requireWitness
    "zero training steps produce empty direct and packaged histories"
    ( null (trainIdentity 0 0.01 positiveParams positiveX positiveTarget)
        && null (trainNetCoalgebra positiveParams positiveX positiveTarget 0.01 0)
    )
  requireWitness
    "PINNED status quo: negative steps remain productive beyond the requested magnitude"
    (length (take 3 (trainIdentity (-1) 0 positiveParams positiveX positiveTarget)) == 3)

  let empty = A.array [0] [] :: A.Array Double
      (emptyLoss, emptyGradient) = mseLoss empty empty
  requireWitness "PINNED status quo: empty MSE loss is NaN" (isNaN emptyLoss)
  requireWitness
    "PINNED status quo: empty MSE gradient has no values"
    (null (arrayAs emptyGradient :: [Double]))

  let (shortLoss, shortGradient) = mseLoss (vec [2, 4]) (vec [1])
      (longLoss, longGradient) = mseLoss (vec [2]) (vec [1, 3])
  requireWitness
    "PINNED status quo: shorter target truncates MSE to 1/2"
    (shortLoss == 0.5)
  requireWitness
    "PINNED status quo: shorter target keeps prediction shape and gradient [1]"
    (A.shape shortGradient == A.shape (vec [2, 4]) && arrayAs shortGradient == [1])
  requireWitness
    "PINNED status quo: longer target truncates MSE to 1"
    (longLoss == 1)
  requireWitness
    "PINNED status quo: longer target keeps prediction shape and gradient [2]"
    (A.shape longGradient == A.shape (vec [2]) && arrayAs longGradient == [2])

  let extraBiasParams =
        netParamsFromArrays
          (A.array [1, 1] [1])
          (A.array [2] [0, 100])
          (A.array [1, 1] [1])
          (A.array [1] [0])
  requireWitness
    "PINNED status quo: an extra bias value is ignored by forward"
    (forward extraBiasParams (vec [1]) == vec [1])

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
