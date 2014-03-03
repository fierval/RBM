namespace Dbl.Rbm

//open MathNet.Numerics.LinearAlgebra
//open MathNet.Numerics.LinearAlgebra.Double
open Fmat.Numerics
open Fmat.Numerics.MatrixFunctions
open Fmat.Numerics.BasicStat
open Fmat.Numerics.LinearAlgebra

open System.Threading
open System.Threading.Tasks
open System.Linq

open Microsoft.FSharp.Linq
open System.Collections.Generic

[<AutoOpen>]
module RBM =
    
    let memberwiseMult (x : Matrix) (y: Matrix)  =
        Matrix.applyFun2Arg (x, y, fun x1 x2 -> x1 * x2)
    
    // uniform random for sampling
    let rand () = (rand [1;1]).[1,1]
    
    // create a vector from a one dimensional array
    let vector (arr : float []) = Matrix(Array2D.init arr.Length 1 (fun i j -> arr.[i]))
    let array (m : Matrix) = m.ToColMajorSeq() |> Seq.toArray

    type Rbm (nHidden, data : (float [] list)) as this = 
        [<DefaultValue>]
        val mutable curModel : Matrix list

        [<DefaultValue>]
        val mutable curModelHidden : Matrix list

        [<DefaultValue>]
        val mutable hiddenProbs : Matrix list

        [<DefaultValue>]
        val mutable hiddenProbsG : Matrix list

        // length of the visible units vector
        let nVisible = if data = Unchecked.defaultof<float [] list> || data.Length = 0 then failwith "0 visible units" else data.First().Length
        let nHidden = if nHidden > 0 then nHidden else failwith "0 hidden units"

        // data - a list of boolean vectors to represent the data
        let mutable data = data |> List.map (fun x -> Matrix(Array2D.init 1 data.Length (fun i j -> data.[i].[j])))

        let alpha = 0.2

        let sigm (x : Matrix) = 1.0 / (1.0 + exp(-x))

        // weight matrix W[j,k]: j - hidden, k - visible (# of units)
        let mutable weights = Matrix.normalRnd(0.0, 1.0, [nVisible; nHidden])
        let mutable prevWeights = weights

        // hidden bias, b[j]
        let mutable hiddenBias = Matrix.normalRnd(0.0, 1.0, [1; nHidden])
        // visible bias, c[k]
        let mutable visibleBias = Matrix.normalRnd(0.0, 1.0, [1; nVisible])

        let probabVector (bias : Matrix) (weights : Matrix) (units : Matrix) =
            sigm(bias + weights * units)
        
        let probab (bias : float) (weights : Matrix) (units : Matrix) =
            sigm(bias + weights * units).[0,0]
 
        // vector of probabilities for each hidden neuron j p(h[j] = 1 | x)
        let probabHiddenGivenVisible (visible : Matrix) = 
            vector (Array.init nHidden (fun j -> probab hiddenBias.[0, j] weights.[j..j, 0..weights.Size.[1] - 1] visible))

        // vector of probabilites for each visible neuron k p(x[k] = 1 | h)
        let probabVisibleGivenHidden (hidden : Matrix) = 
            vector (Array.init nVisible (fun  k -> probab visibleBias.[0, k] weights.[0..weights.Size.[0] - 1, k..k].T hidden))

        /// Update the weights & biases:
        /// For each training example, t:
        /// W <= W + alpha(h(x)x.T - h(xG)xG.T)
        /// b <= b + alpha(h(x) - h(xG))
        /// c <= c + alpah(x - xG)
        let updateParameters t =
            let x = data.[t]
            let xG = this.curModel.[t]
            let hx = this.hiddenProbs.[t]
            let hxG = this.hiddenProbsG.[t]

            weights <- weights + alpha * (hx * x.T - hxG * xG.T)
            hiddenBias <- hiddenBias + alpha * (hx - hxG)
            visibleBias <- visibleBias + alpha * (x - xG)
                        
        // perform Gibbs sampling for the number of steps,
        // starting with the 
        let bernoulliGibbsSample steps =
            let oneLayerToAnother nAnotherUnits (probs : Matrix list) (d : Matrix list) = 
                d 
                |> List.mapi (fun k e ->
                    let uniformSample = [|for i = 1 to e.Length do yield rand()|]
                    let newVector = Array.fold2 (fun state rnd p -> if p > rnd then state @ [1.0] else state @ [0.0]) List.empty uniformSample (array probs.[k])
                    let toMatrix = Matrix(Array2D.init 1 nAnotherUnits (fun i j -> newVector.[j]))
                    toMatrix
                    )

            let visibleToHidden = oneLayerToAnother nHidden
            let hiddenToVisible = oneLayerToAnother nVisible

            let rec performSteps visible hidden hiddenProbs step = 

                if step = 0 then visible, hidden, hiddenProbs
                else
                    let hidden = visibleToHidden hiddenProbs visible

                    let visibleProbs = hidden |> List.map(fun e -> probabVisibleGivenHidden e)
                    let visible = hiddenToVisible visibleProbs hidden

                    let hiddenProbsG = visible |> List.map(fun e -> probabHiddenGivenVisible e)
                    let hidden =  visibleToHidden hiddenProbsG visible
                    performSteps visible hidden hiddenProbsG (step - 1) 
 
            this.hiddenProbs <- data |> List.map (fun e -> probabHiddenGivenVisible e)

            let model, modelHidden, modelHiddenProbsG = performSteps data Unchecked.defaultof<Matrix list> this.hiddenProbs steps

            // save the values we need to update parameters
            this.curModel <- model
            this.curModelHidden <- modelHidden
            this.hiddenProbsG <- modelHiddenProbsG
        
        /// error is the square sum of differences between
        /// last epoch weights and current weights
        let computeError () =
            let mutable errorMatrix = weights - prevWeights
            
            errorMatrix.ApplyFun(fun e -> e * e)
            Matrix.sum(Matrix.sum(errorMatrix, 0), 1).[0,0]

        /// Run training for the number of epochs or when the error
        /// is less than the error specified. error < 0 means ignore error
        member this.Train epochs error =
            let mutable stop = false
            let rec train remainEpochs curError =
                if remainEpochs = 0 || (error > 0.0 && curError <= error) then
                    ()
                else
                    bernoulliGibbsSample 1
                    /// store current weights for error update
                    prevWeights <- weights
                    for i = 0 to data.Length - 1 do
                        updateParameters i

                    /// update error
                    let curError = if error < 0.0 then error else computeError ()
                    train (remainEpochs - 1) curError
            train epochs error