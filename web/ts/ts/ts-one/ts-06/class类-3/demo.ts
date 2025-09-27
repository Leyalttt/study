class Animall {
  name: string = '动物'
  sleep() {
    console.log('动物在睡觉')
  }
}

class Dog2 extends Animall {
  name: string = '狗'
  sleep() {
    console.log('狗在睡觉')
  }
}

let dog2 = new Dog2()
console.log(dog2.name)
dog2.sleep()