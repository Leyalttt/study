//1.当做类型声明去使用
interface myInterface {
  name: string,
  age: number,
  sleep(): void
}
interface myInterface {
  gender: string
}
let a: myInterface
a = {
  name: '张三',
  age: 18,
  sleep() {},
  gender: '雄性'
}

//类的类型声明使用
class Dog implements myInterface {
  name: string
  age: number
  gender: string
  sleep(): void {
    console.log(this.name + '在休息')
  }
  constructor(name: string, age: number, gender: string) {
    this.name = name
    this.age = age
    this.gender = gender
  }
}
let dog = new Dog('金毛', 2, '男')
dog.sleep()