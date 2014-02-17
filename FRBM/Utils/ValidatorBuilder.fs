namespace Utils

open System
open System.IO

type Arg =
    | Str of string
    | FileName of string

[<AutoOpen>]
module ValidatorModule =
    type ValidatorBuilder() =
        let isValid arg = 
            match arg with
            | Str s -> if System.String.IsNullOrWhiteSpace(s) then failwith "String cannot be empty" else s
            | FileName f -> if File.Exists f then f else failwith ("File does not exist: " + f)

        member this.Bind(arg, passOn) : Arg =
            let valid = isValid arg
            passOn valid 
        
        member this.ReturnFrom (x : Arg) : Arg = Str(isValid x)
        member this.Delay func = func()
        member this.Run (x : Arg) = 
            match x with
            | Str s | FileName s -> s

    let valid_arg = new ValidatorBuilder()