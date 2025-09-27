abstract class Anmial {
  //抽象方法没有方法体
  abstract sleep(): void
}

//抽象类不能被实例化
// let an = new Anmial() 会报错误

//抽象类被继承, 抽象方法需要重写
// class Dog3 extends Anmial {
//   sleep() {

//   }
// }

//抽象类继承抽象类
abstract class Dog3 extends Anmial {
 abstract eat(): void
}
 class JM extends Dog3 {
  sleep(): void {
    
  }
  eat(): void {
    
  }
 }
