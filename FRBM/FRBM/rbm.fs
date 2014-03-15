namespace Dbl.Rbm

//open MathNet.Numerics.LinearAlgebra
//open MathNet.Numerics.LinearAlgebra.Double
open Fmat.Numerics
open Fmat.Numerics.MatrixFunctions
open Fmat.Numerics.BasicStat
open Fmat.Numerics.LinearAlgebra

open System
open System.Threading
open System.Threading.Tasks
open System.Linq

open Microsoft.FSharp.Linq
open System.Collections.Generic

[<AutoOpen>]
module RBM =
    
    type Rbm (nHidden, data : (float [,])) = 

        // length of the visible units vector
        let nVisible = if data = Unchecked.defaultof<float [,]> || data.Length = 0 then failwith "0 visible units" else Array2D.length2 data
        let nHidden = if nHidden > 0 then nHidden else failwith "0 hidden units"

        // data - a list of boolean vectors to represent the data
        // each case is a column
        let mutable origData = Matrix(data).T
        let mutable curData = Matrix(data).T

        // Hinton recommends batch size 10 - 100.
        let batchSize = if origData.Size.[0] <= 10 then origData.Size.[0] else Math.Min(origData.Size.[0] / 10, 100)

        let alpha = 0.2

        let sigm (x : Matrix) = 1.0 / (1.0 + exp(-x))

        // weight matrix W[j,k]: j - hidden, k - visible (# of units)
        let mutable weights = Matrix.normalRnd(0.0, 1.0, [nVisible; nHidden])
        let mutable prevWeights = zeros [nVisible; nHidden]

        let probabMatrix (weights : Matrix) (units : Matrix) =
            sigm weights * units
        
        // vector of probabilities for each hidden neuron j p(h[j] = 1 | x)
        let probabHiddenGivenVisible = probabMatrix weights

        // vector of probabilites for each visible neuron k p(x[k] = 1 | h)
        let probabVisibleGivenHidden = probabMatrix weights.T

        // perform Gibbs sampling for the number of steps,
        // starting with the 
        let trainSingleStep () =
            // stochastically pick 0 or 1 based on the probability matrix
            let stochasticBinarize (probs : Matrix) =
                let uniform = rand [batchSize; nHidden]
                Matrix.applyFun2Arg(probs, uniform, fun p u -> if p >= u then 1.0 else 0.0)                

            for i in 0..batchSize..origData.Size.[1] - 1 do
                let finish = Math.Min(i + batchSize - 1, origData.Size.[1] - 1)
                let visible = curData.[0..nVisible - 1, i..finish]
                let probabHidden = probabHiddenGivenVisible visible

                // per Hinton, this layer of hidden is set to 0's and 1's
                let hidden = stochasticBinarize probabHidden

                // Go down
                // It's ok to keep probabilities and not set values to "0" and "1" for this layer
                // (Hinton, section 3.2). Since, however, persistent version of CD is used, this may
                // make a difference (?)
                let visible' = probabVisibleGivenHidden hidden |> stochasticBinarize

                // Go up again
                // Not setting to "0" or "1", since nothing depends on them anymore
                let hidden' = probabHiddenGivenVisible visible' //no need to transpose visible here.

                let deltaWeights = (hidden * visible.T - hidden' * visible'.T) / float (finish - i + 1)
                weights <- weights + alpha * deltaWeights
  
                // persist the computed visible
                curData.[0..nVisible - 1, i..finish] <- visible'


        /// error is the square sum of differences between
        /// last epoch weights and current weights
        let computeError () =
            let errorMatrix = weights - prevWeights
            
            errorMatrix.ApplyFun(fun e -> e * e)
            Matrix.sum(Matrix.sum(errorMatrix, 0), 1).[0]

        /// Run training for the number of epochs or when the error
        /// is less than the error specified. error < 0 means ignore error
        member this.Train epochs error =
            let rec train remainEpochs curError =
                if remainEpochs = 0 || (error > 0.0 && curError <= error) then
                    ()
                else
                    // store current weights for error update
                    prevWeights <- Matrix.applyFun(weights, fun e -> e)
                    
                    // run a step of training
                    trainSingleStep ()

                    // update error. Don't update it if curError < 0 (we are not interested in error computations)
                    let curError = if error < 0.0 then error else computeError ()

                    train (remainEpochs - 1) curError
            train epochs error