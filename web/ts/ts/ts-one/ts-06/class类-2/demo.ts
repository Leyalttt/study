class Animal {
  name: string
  constructor(name: string) {
    this.name = name
  }
  sleep() {
    console.log(this.name + '在睡觉')
  }
}

class Dogg extends Animal {
  constructor(name: string) {
    super(name)
  }
}

let dogg = new Dogg('打光')
dogg.sleep()