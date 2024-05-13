>
    .
        >>
            ..
                >>>
                    ...

A
    I)
         a)
               a1)

>
    §
       Σ
          ∂

rule root = element * { }
rule element = (tag / data) *(element / data)
rule tag = "@" char +
rule data = char +
expression ::= term
             | expression "+" term
             | expression "-" term

term       ::= factor
             | term "*" factor
             | term "/" factor

factor     ::= "(" expression ")"
             | number

number     ::= digit
             | digit number

digit      ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
