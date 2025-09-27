//1.类型不明确就是用泛型
function get(name: string): string {
  return name
}

function getNum(age: number): number {
  return age
}

// function getAny(a: any): any {
//   let num: number = 456
//   num = a
//   return num
// }

// let str = getAny('str')
// console.log(typeof str) //string

function getAny(a: unknown): unknown {
  let num: number = 456
  num = a as number
  return num
}

let str = getAny('str')
console.log(typeof str) //string


//泛型: 泛指一类型
function getFx<T>(a: T): T {
  return a
}
let str2 = getFx("hello") //str2就是string类型
let nu = getFx(123)//nu就是number类型

//泛型的约束
interface P {
  name: string,
  age: number
}

function getPerson<T extends P>(arg: T): T {
  return arg
}

getPerson({
  name: "张三",
  age: 18
})

//创建多个泛型
function getMsg<S, N>(name: S, age: N): [S, N] {
  return [name, age]
}
getMsg('zs', 18)

//
interface Anmail<G> {
  gender: G
}

class Dog5<S, N, G> implements Anmail<G> {
  name: S
  age: N
  gender: G
  constructor(name: S, age: N, gender: G) {
    this.name = name
    this.age = age
    this.gender = gender
  }
}

let dog2 = new Dog5("大黄", 15, '雄性')