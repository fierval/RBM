namespace Dbl

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

/// Neural net with back propagation and weight decay regularization, using sigmoid activation function
/// <parameter name = "data"> Data set, individual observations in rows</parameter>
/// <parameter name = "labels"> A vector of labels for each observation. 1[i==c] (1 - identity function) </parameter>
/// <parameter name = "hiddenLengths> Lenghts of hidden layers </parameter>
/// <parameter name = error> Squared error determining where to stop. default value: 0.05 </parameter>
/// <parameter name = lambda> Regularization constant for weight decay. If not specified set to 0 (no regularization) </parameter>
type NeuralNetwork(data : float [,], labels : float [], hiddenLengths : int seq, ?error, ?lambda) =
    let lambda = defaultArg lambda 0.0
    let error = defaultArg error 0.05

    let nVisible = if data = Unchecked.defaultof<float [,]> || data.Length = 0 then failwith "0 visible units" else Array2D.length2 data
    let totalSamples = Array2D.length1 data
    
    let hiddenLengths = hiddenLengths |> Seq.toList

    // not pretty, but does the job...
    // need to add an extra column for "1's"
    let input = concat([Matrix(data); Matrix(Array2D.create totalSamples 1 1.0)], 1)
        
    // create layers (column vectors)
    let layers = 
        [|
            for len in hiddenLengths -> Matrix(Array.init (len + 1) (fun i -> if i < len then 0.0 else 1.0)).T
            let arr : float [] = Array.zeroCreate labels.Length
            yield Matrix(arr).T
        |]

    //index of the output layer - comes in handy
    let outLayer = layers.Length - 1        

    // create and initialize weight matrices
    let weights =
        [|
            yield! layers 
            |> Seq.mapi 
                (fun i layer -> 
                    let rows = if i = 0 then nVisible + 1 else layers.[i - 1].Size.[0]
                    let cols = layer.Size.[0]
                    Matrix.normalRnd(0.0, 1.0, [rows;cols])
                )
        |]

    // sigmoid activation
    let activation x = 1.0 / (1.0 + exp(-x))
    let activation' x = 
        let y = activation x
        y * (1.0 - y)

    // given the result vector t, compute squared error = 1/2 * Sum(t[i] - y[i])
    let sqError (t : Matrix) =
        let diffOutResult = t - layers.[outLayer]
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