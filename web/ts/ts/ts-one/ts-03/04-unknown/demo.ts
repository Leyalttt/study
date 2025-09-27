let un: unknown
un = 123
un = 'unun'
un = true

let c: unknown
c = 'ccc'
let d: string

if (typeof c === 'string') {
  d = c
}

//unknown 看成安全的any类型