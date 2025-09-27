class Person {
  name: string = '张三'
  static age: number = 18
  static sayHello() {
    console.log('你好')
  }
  readonly sex:string = '女'
}

// let person = new Person()
// console.log(person.name)
// console.log(Person.age)
// console.log(Person.sayHello())
// console.log(person.sex)
/**
 * 被 static 修饰的属性或者是方法，属于类的。
  可以通过类名调用，不属于实例的，实例没办
  使用
 */

class Dog {
  name: string
  constructor(name: string) {
    this.name = name
  }
}
//当我们调 new Dog();的时候我们就等于调用 Dog 中的构造方法
let dog = new Dog('中华田园犬')
console.log(dog.name)
