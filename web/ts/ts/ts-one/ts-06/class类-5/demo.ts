class Person2 {
  // public name: string
  // protected name: string
  //私有的, 只能在自己里访问
  private name: string
  constructor(name: string) {
    this.name = name
  }
  // public getName() {
  //   console.log(this.name)
  // }
  // protected getName() {
  //   console.log(this.name)
  // }
  private getName() {
    console.log(this.name)
  }
  getAge() {
    this.getName()
  }
}

// protected只能在内部和继承里访问
class Man extends Person2 {
  public getName(): void {
    console.log(this.name)
  }
  getAge() {
    this.getName()
  }
}

//public 在外部和内部都可以访问
let person = new Person2('张三')
// person.getName()
// console.log(person.name)