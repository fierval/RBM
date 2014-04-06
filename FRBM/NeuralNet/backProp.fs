﻿namespace Dbl

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
module NeuralNetsModule =
    /// Neural net with back propagation and weight decay regularization, using sigmoid activation function
    /// <parameter name = "data"> Data set, individual observations in rows</parameter>
    /// <parameter name = "labels"> A matrix of labels for each observation. For ith observation: 1[i, j==c] (1 - identity function) </parameter>
    /// <parameter name = "hiddenLengths> Lenghts of hidden layers </parameter>
    /// <parameter name = error> Squared error determining where to stop. default value: 0.05 </parameter>
    /// <parameter name = lambda> Regularization constant for weight decay. If not specified set to 0 (no regularization) </parameter>
    type NeuralNetwork(data : float [,], labels : float [,], hiddenLengths : int seq, ?error, ?lambda) =
        let lambda = defaultArg lambda 0.0
        let error = defaultArg error 0.05
        // learing rate
        let gamma = 0.02

        let nVisible = if data = Unchecked.defaultof<float [,]> || data.Length = 0 then failwith "0 visible units" else Array2D.length2 data
        let totalSamples = Array2D.length1 data
    
        let hiddenLengths = hiddenLengths |> Seq.toList

        // not pretty, but does the job...
        // need to add an extra column for "1's"
        let input = concat([Matrix(data); Matrix(Array2D.create totalSamples 1 1.0)], 1)
        let output = Matrix(labels)
        
        // create layers (column vectors)
        let layers = 
            [|
                // for each hidden layer, we add 1 at the end
                yield input
                for len in hiddenLengths -> Matrix(Array.init (len + 1) (fun i -> if i < len then 0.0 else 1.0)).T
                let arr : float [] = Array.zeroCreate (Array2D.length2 labels)
                yield Matrix(arr).T
            |]

        // index of the output layer - comes in handy
        let outLayerIndex = layers.Length - 1 
        // output layer itself       
        let outLayer = layers.[outLayerIndex]

        // create and initialize weight matrices
        let weights =
            [|
                yield! layers |> Seq.skip 1
                |> Seq.mapi 
                    (fun i layer -> 
                        let rows = layers.[i].Size.[0]
                        let cols = layer.Size.[0]
                        Matrix.normalRnd(0.0, 1.0, [rows;cols])
                    )
            |]

        // sigmoid activation
        let activation (x : Matrix) = 1.0 / (1.0 + exp(-x))
        let activation' (x : Matrix) = 
            let y = activation x
            y * (1.0 - y)

        // given the result vector t, compute squared error = 1/2 * Sum(t[i] - y[i])
        let squareError (t : Matrix) =
            let diffOutResult = t - layers.[outLayerIndex]
            let error = 0.5 * (diffOutResult.T * diffOutResult)
            // regularization contribution
            if lambda > 0.0 then
                let squares = 
                    [
                        for weight in weights do
                            let sq = Matrix.applyFun(weight, fun w -> w * w)
                            yield (sum(sum(sq, 0), 1)).[0,0]
                    ]
                lambda * (1.0 / float totalSamples) * squares.Sum() + error
            else
                error

        // forward propagation step given a zero-based n+1 dimensional input vector
        // (where x[n] = 1)
        let forwardProp j =

            // input vector has already been modified to contain 1.0 in the last position
            layers.[0] <- input.[j..j, 0..input.Size.[1] - 1]

            for i = 0 to layers.Length - 1 do
                layers.[i + 1] <- activation (layers.[i] * weights.[i])
                // there is no bias neuron in the output layer
                if i <> outLayerIndex then
                    layers.[i + 1].[0, layers.[i + 1].Size.[1] - 1] <- 1.0

        // back-propogate.
        // first - the output layer, then - the rest of them
        // j - jth input
        let backProp j =
        
            let errorVector = (outLayer - output.[j..j, 0..output.Size.[1] - 1]).T

            // diagonal matrices of derivates
            // we don't take the last weight since it doesn't play any part in setting the bias neuron
            let derivatives =
                [
                    for i = 0 to outLayerIndex do
                        let len = if i = outLayerIndex then outLayer.Length else layers.[i].Size.[1] - 1
                        yield diag (Matrix([1; len], fun i -> (activation' layers.[i]).[0,0]), 0)
                ]                

            // compute error matrices
            let deltas = List<Matrix>()
            deltas.Add(derivatives.[outLayerIndex] * errorVector)

            for i = weights.Length - 1 to 0 do
                // this ugly [..] range expression means we are chopping off the final column of weights, while taking all the rows.
                let delta = derivatives.[i] * weights.[i].[0..weights.[i].Size.[0] - 1, 0..weights.[i].Size.[1] - 2] * deltas.[deltas.Count - 1]
                deltas.Add(delta)

            deltas.Reverse()

            // compute deltas of weights and adjust the weights
            for i = weights.Length - 1 to 0 do
                let deltaW = (-gamma * deltas.[i] * layers.[i]).T
                weights.[i] <- weights.[i] + deltaW
                        


