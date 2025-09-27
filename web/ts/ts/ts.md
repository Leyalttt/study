### 第 1 集：带你认识 TypeScript

**简介：介绍新朋友 TypeScripe**

![img](./image/1.png)

1、TypeScript 是什么呢？

````
1）：JavaScript：动态类型语言（Dynamically Typed Language），在运行期间检查数据的类型的语言。用这类语言编程，不会给变量指定类型，而是在赋值时得到数据类型。
例如：
    案例一：
    var a=123;
    a='ndedu';
    a=true;


    案例二：

    function fun(name){

        console.log(name);	·
    }


fun();
fun('ndedu',123);

### 第2集：TypeScript环境搭建

**简介：TypeScript环境的安装**



1、安装node.js

- node.js下载地址

  - 地址: https://nodejs.org/en/download/

- 安装

- 测试是否安装成功

  - ```
    进入命令行：
```

    ```
        node -v
    ```
    
    ```
    
    ```

- npm配置国内镜像源

  - ```
    这里使用的是淘宝的镜像：
    ```

    ```
    npm config set registry https://registry.npmmirror.com
    ```

    ```
    配置之后可以验证是否成功（返回刚刚设置的地址即成功）：
    ```

    ```
    npm config get registry
    ```



2、使用npm全局安装typescript

````

1.进入 DOS 界面 2.输入：npm install -g typescript 3.验证：tsc -v

```



 3 全局安装后的目录在哪里？

npm config get prefix



### 第3集：开始我们第一个 TS程序

**简介：带领大家体验一下编写一个ts程序**



1、安装vscode

```

vscode 官网下载地址：https://code.visualstudio.com/

```



2、将ts文件编译成js文件

```

tsc 被编译文件的名字.ts
终端输入tsc hello.ts 会编译成js文件

tsc 的意思是
TypeScript Complie

````

3. 然后运行js文件 如:node demo.js



## 第三章 详解TypeScript中简单的数据类型

### 第1集：类型声明

**简介：TS中的类型声明**



- 类型声明

  - 类型声明是TS中非常重要的特点

  - 通过类型声明可以指定TS中变量（形参）的类型

  - 指定类型后，当变量赋值时，TS编译器就会自动检查值是否符合声明类型，如果符合就赋值，不符合就报错

  - 简单的说，类型声明就是给变量设置了类型。使得变量只能存储某种类型的值。

  - 语法：

    ```
    类型声明：
    ```

    ```
        let 变量名:类型;//只声明，未赋值
    ```

    ```
        let 变量名:类型=值;//声明并赋值
    ```

    ```
        function fun(参数:类型,参数:类型):类型{
    ```

    ```

    ```

    ```
        }
    ```

    ```

    ```

    ```

    ```

    ```

    ```









### 第2集：自动类型判断

**简介：讲解TS中的自动类型判断**

- 什么是自动类型判断？
  - TS拥有自动类型判断机制
  - 当对变量的声明和赋值同时进行时，TS编译器会自动判断变量类型
  - 所以如果你的变量的声明和赋值是同时进行的，可以省略掉类型声明



- 对应的案例

  - ```
    let one:number;//声明未赋值
    ```

    ```
    one=7;
    ```

    ```
    let two=8;// 声明一个变量并赋值
    ```

    ```

    ```

    ```
    two='ndedu';//此时会提示报错
    ```

    ```
    console.log(typeof one);
    ```

    ```
    console.log(typeof two);
    ```





### 第3集：字面量声明

**简介：详解TS中字面量声明**



1、字面量

````

字面量：是个客观存在无法改变的值。  
 例如： 1 2 3 4 5 6 true 'hello'

```

2、字面量声明的

```

let a:23;
使用字面量声明后 a 的值，永远是 23 相当于常量不能修改
const a=23;
如果修改 ts 就会提

```







### 第4集：TS中any类型

**简介：详解any类型的细节**

1、声明any类型

```

any:表示任意数据类型，一个变量设置类型为 any 后相当于对该变量关闭了 TS 的类型检测
let a:any;//显式的 any 类型声明
a=12;
a=true;
a="ndedu";

let b;//隐私的 any 类型声明，声明如果不指定类型，则 TS 解析器会自动判断变量的类型为 any
b=23;
b=false;
b="ndedu";

```



2、使用any存在的隐患

```

使用 any 声明的变量它不仅影响自己，同时还影响别人

```



### 第5集：TS中unknown类型

**简介：详解unknown类型的细节**



1、声明unknow类型

```

unknown:表示未知类型的值;
let b:unknown;
b=123;
b=true;
b='ndedu';

let c:unknown;
c='123';
let d:string;
d=c;//此时 TS 解析器提示是报错的
//虽然 变量中的字面量都是 string，但是 d 是 string 类型 c 是 unknown 所以不能赋值
//假如我就想让 c 的值赋值给 d,该怎么操作呢？

//这个也可以实现
//我们需要先判断
if(typeof c ==='string'){
d=c;//这样就可以完成赋值了
}

```

2、unknown可看成安全的any





### 第6集：TS中的类型断言

**简介：详解TS中的类型断言**



类型断言：可以用来告诉TS解析器变量的实际类型

语法规则：

```

语法：
变量 as 类型;
<string> 变量;
利用类型断言来处理上一节的 unknown 赋值的小问题
let c:string;
let d:unknown;
d="ndedu";
c=d as 'string';
ok 解决问题

```





### 第7集：TS中的函数

**简介：详解TS中的函数定义的细节**

1、对函数中形参进行类型声明

```

如何设置函数声明中的形参的类型声明呢？
语法：
function fun(形参 1:number,形参 2:number){
console.log(形参 1+形参 2);
}
//调用
//传递不是 number 的实参
fun('ndedu','web');//TS 解析器提示报错，不允许这样做，我们必须按照函数声明的形参类型传递参数
//正确的参数
fun(123,456);
//多参数
fun(123,4556,55,56);//TS 解析器报错
//少参数
fun(11);//TS 解析器报错

```

2、函数中返回值类型的声明

```

第一种情况：不设置返回值类型，但是有返回值
function fun(){
return true;
}
这个时候 TS 解析器会根据返回值的类型进行判断，返回值类型是什么函数返回值类型就是什么
let result=fun();
console.log(typeof result);

第二种情况：设置返回值类型为 void
void:用来表示空，以函数为例，就表示没有返回值的函数
function fun():void{
return 'a';//提示报错
return 123;//提示报错
return true;//提示报错
return undefined;//不提示报错
return null;//不提示报错
return;//不提示报错
}

第三种情况：设置返回值类型为 never
never:永远不会返回结果

function fun():never{
throw new Error("出错了！！");
//程序报错代码立即结束，立即结束以后就不会有返回值了,这个东西用的相对比较少，了解一下知道一下就行了
}

```







## 第四章：带你快速掌握TS中的复杂的数据类型声明

### 第1集：TS中的对象类型声明

**简介：详解讲解TS中对象类型声明的知识**



```

let a:object; //这就表示声明一个变量，类型为 object,也就是说 a 只能用来报存一个对象
//object 表示一个对象

a={};
a=function(){};
//这样用不实用，因为在 js 中对象太多了，一切皆为对象。
//我们更多的想限制对象中有哪些属性

let b:{name:string};//这样写表示声明一个变量 b 类型为对象，对象中只能有一个属性为名为 name 且属性值的类型为 string

b={};//提示报错
b={name:'张三'};//这样写 才正确

用来指定对象中可以包含哪些属性
语法：
{属性名:属性值，属性名 1:属性值 1}
注意当我们通过这种方式指定对象的时候，你写的这个对象必须要这个要求指定对象的格式必须一模一样

    情景一：我们指定的对象它必须包含一个name,但是还有age和sex这个俩个变量是可能有可能有没有的，这种情况我们该怎样做呢？

这种情况下我我们在属性名后面加个?，它就表示这个属性可选，既可以有也可以无；

例如
let b:{name:string,age?:number,sex?:string};

b={name:'张三'};
b={name:'张三',age:12};

情景二：当我们只知道我们必须有的属性，而其他的不必须我们不知道，我们该怎么处理呢？
我们可以通过下面这个方式实现
let b:{name:string,[propName:string]:unknown};
//这就表示我们指定 b 中存储的类型是对象，并且这个对象必须含有一个 name 的属性，类型为 string,这个对象还可以有其他可选属性，只要属性名满足是字符串，属性值是 unknown 即可
[propName:string]:
propName:这个任意命名，就表示属性的名字，这个属性名字的类型是 string。js 中属性名一般都是用 string 定义
[propName:string]:unknown;合起来就代表属性名为 string 类型，为可选的并且这个属性的值为 unknown 类型

```





### 第2集：TS中数组类型声明

**简介：详细讲解TS中数组类型的声明**

```

在 TS 我们可以通过
let e:string[];//声明一个数组
string[];//表示字符串数组
e=['a','b','ndedu',1];//但是你存一个 TS 解析器就是提示报错
再创建一个 number 型的数组
let d:number[];
boolean 型的数组
let e:boolean[];

另外一种数组的声明方式

let e:Array<number>;

数组类型声明的语法：
第一种：
let 变量名:类型名[];
第二种：
let 变量名:Array<类型名>;

假如你还想创建一个能存储任意类型的数组。怎么办呢？
可以这样做
let e:Array<any>;
let e:any[];

```





### 第3集：TS中的扩展类型tuple

**简介：详解TS中的扩展类型tuple**

1、tuple：叫做元组

2、什么是元组呢？

​	元组就是定长的数组。（就代表数组中的数量是固定的就叫做元组，元组的存储效率比较好点，因为元组是固定，不会出现扩容的现象，所有效率会好点）

​	使用场景：

​		就是数组的数量是固定的，这时候用元组是比较好的选择。

3、元组怎么写呢？

```

let h:[string,string];//这就表示我定义一个元组，在这个元组中有两个，第一个值是 string 类型。第二个值也是 string
//定义的时候多一个少一个也不行，必须按照声明的结构定义数组，不然 TS 解析器就会提示报错

h=['ndedu','net'];
元组书写语法：[类型，类型,类型]；//这个不会特别长，元素多的话还是用数组

ok 这就是 TS 中的元组

```





### 第4集：TS中Enum

**简介：详解TS中Enum**

1、什么是enum（枚举）?

```

在数学和计算机科学理论中，一个集的枚举是列出某些有穷序列集的所有成员的程序，或者是一种特定类型对象的计数

```

2、TS中的枚举Enum

```

使用枚举，可以定义一些带名字的常量。用于清晰地表达意图或创建一组有区别的用例。

语法：
enum 枚举名称{成员 1，成员 2....};

1）：数字枚举
默认情况下，第一个枚举值是 0，后续至依次增 1
enum Color
{
red,
blue,
yellow
}

    let col = Color.blue;
    alert(col);
    // 1

2）：字符串枚举

    enum gender {
        male = '1',
        female = '0',
    }

    alert(gender.male);
    // "1"

```





### 第5集：TS中联合类型声明

**简介：给大家讲解如何在ts进行联合类型声明**

```

在 ts 中我们可以使用 "|" 进行联合类型声明

语法：
let 变量名:声明类型 1|声明类型 2.....;
//可以是任意多个

    //就表示声明一个变量名，它可以是类型1还可以是类型2

```





### 第6集：TS中类型别名

**简介：详解讲解TS中的类型别名**

```

什么叫类型别名呢？
比如：
let k= 1|2|3|4|5|;
我这里还有一个值 p 的范围和 k 是一样的
let p=1|2|3|4|5|;
假如他有几十个呢？我们这样写岂不是麻烦。
这个时候我们可以使用 TS 提供的类型别名

    语法：
      type 类型别名的名字=string;

    现在可以看做myType和string是等价的
    我们测试一下
    let b:string;
    let c:myType;

    //现在我们打印一下 类型是否相等
    console.log(typeof b=== typeof c);


    我们接下来使用myType来简化我们之前这个案例
    type myType=1|2|3|4|5|;
    let k=myType;
    let p=myType;
    //这样是不是方便多了，如果值有很多那可以大大简化我们书写的代码

```





## 第五章 详解TS中的配置

### 第1集：自动编译文件

**简介：讲解如何开启自动编译ts文件的功能**



1、自动编译单个文件

```

编译文件时，使用-w 指令后，TS 编译器会自动监视文件的变化，并在文件发生改变时对文件进行重新编译。
tsc 文件名.ts -w

```



2、自动编译当前项目下的所有的ts文件

```

如果直接使用 tsc 指令，则可以自动将当前项目下的所有的 ts 文件编译成 js 文件
有一个前提，首先在项目的根目录下创建一个 ts 的配置文件 tsconfig.json
tsconfig.json 是一个 json 文件，添加配置后，只需要 tsc 命令即可完成对整个项目的编译

```



### 第2集：详解tsconfig.json中的配置选项

**简介：本节课主要给大家讲解tsconfig.json中的4个配置选项**



1、include

```

定义希望被编译文件所有的目录
默认值：[**/**]
案例：
"include":["dev/**/*","prop/**/*"]

     **：表示任意目录
     *：表示任意文件

     //就代表所有的dev目录和prop目录下的文件都会被编译

```



2、exclude

```

定义不需要编译的目录
默认值：["node_modules","bower_components","jspm_packages"]
案例："exclude":["./prop/**/*"]
//代表不编译 prop 目录下的所有文件

```



3、extends

```

定义被继承的配置文件
案例：
"extends":"./config/base"

    //表示当前配置文件会自动包含config目录下base.json中的所有配置信息

```



4、files

```

指定需要编译文件的列表
案例：
"files":
[
"one.ts",
"two.ts",
"three.ts",
"four.ts"
]

    //  表示列表中文件都会被TS编译器编译

```



5、complierOptions：ts编译时的配置



### 第3集：详解tsconfig.json中的complierOptions

**简介：详细讲解complierOptions子选项中的配置**



1、target

```

指定 ts 编译的 js 目标版本
可选值："ES3"（默认）， "ES5"， "ES6"/ "ES2015"， "ES2016"， "ES2017"或 "ESNext"。
案例：
"compilerOptions":{
"target":"ES6"
}

    //表示我们所编写的ts代码将会被编译ES6版本的js代码

```

2、module

```

指定使用的模块化规范：
可选值：
"None"， "CommonJS"， "AMD"， "System"， "UMD"， "ES6"或 "ES2015"。
► 只有 "AMD"和 "System"能和 --outFile 一起使用。
► "ES6"和 "ES2015"可使用在目标输出为 "ES5"或更低的情况下。

```



3、lib

```

指定编译过程中需要引入的库文件的列表
可选值：
► ES5
► ES6
► ES2015
► ES7
► ES2016
► ES2017
► ES2018
► ESNext
► DOM
► DOM.Iterable
► WebWorker
► ScriptHost
► ES2015.Core
► ES2015.Collection
► ES2015.Generator
► ES2015.Iterable
► ES2015.Promise
► ES2015.Proxy
► ES2015.Reflect
► ES2015.Symbol
► ES2015.Symbol.WellKnown
► ES2016.Array.Include
► ES2017.object
► ES2017.Intl
► ES2017.SharedMemory
► ES2017.String
► ES2017.TypedArrays
► ES2018.Intl
► ES2018.Promise
► ES2018.RegExp
► ESNext.AsyncIterable
► ESNext.Array
► ESNext.Intl
► ESNext.Symbol

案例：
"compilerOptions":{
"target":"ES6",
"lib":["ES6","DOM"]
}
注意：如果--lib 没有指定默认注入的库的列表。默认注入的库为：
► 针对于--target ES5：DOM，ES5，ScriptHost
► 针对于--target ES6：DOM，ES6，DOM.Iterable，ScriptHost

```





4、outDir

```

用来指定编译后文件所在的目录
案例：
"compilerOptions":{
"target":"ES6",
"lib":["ES6","DOM"],
"outDir":'./dist'
}

```



5、outFile

```

将编译的代码合并成一个文件
案例：
"compilerOptions":{
"target":"ES6",
"lib":["ES6","DOM"],
"outDir":'./dist',
"outFile":'./dist/main.js'
}

"outFile":'./dist/main.js'
//就表示把编译后的文件合并的 main.js 这文件中，最后只会输出一个 js 文件

出现问题的解决方案：

    mudule只能使用 amd或者是system

其实 outFile 这个功能我们在工作中也很少使用，因为我们打包的时候都是通过打包工具进行使用的。所以这个 outFile 大家知道有这个东西了解一下就行

```





### 第4集：详解tsconfig.json中的complierOptions（2）

**简介：详细讲解complierOptions子选项中的配置**

1、allowJs

```

是否对 js 文件进行编译
默认值:false
案例：
"allowJs":true

```



2、checkJs

```

是否检查 js 代码符合语法规范
默认值:false
案例：
"checkJs":true

```



3、removeComments

```

在编译的时候是否移除注释
默认值:false

```



4、noEmit

```

不生成编译后的文件
默认值为：false
案例：
"noEmit":true

```



5、noEmitOnError

```

当存在语法错误的时不生成编译后的文件
默认值：false
案例：
"noEmitOnError":true

```







### 第5集：详解tsconfig.json中的complierOptions（3）

**简介：详细讲解complierOptions子选项中的配置**



1、alwaysStrict

```

js 中有一种模式叫做严格默认，它语法更严格，在浏览器执行的性能更高，开发时候我们都会让我们的代码在严格模式下执行

如果是在 js 文件的话 只需要在 js 文件的开头加入一个字符串
"use strict";
//就表示开启了 js 的严格模式

设置编译后的文件是否使用严格模式
默认值：false
案例：
"alwaysStrict":true

当代码中有引入导出模块代码时，js 会自动进入严格模式下

```



2、noImplicitAny

```

不允许隐式的 any 类型
默认值为：false
案例：
"noImplicitAny":true

```



3、strictNullChecks

```

严格的检查空值
默认值为:false
案例：
"strictNullChecks":false

```



4、strict

```

所有严格检查的总开关
默认值为:false
案例：
"strict":false

```







## 第六章 TypeScript语法进阶



### 第1集：配置在直接运行ts环境



**简介：详解如何直接运行ts文件**

直接运行ts文件，不会生成js文件



1. 首先全局安装ts-node



```

npm install -g ts-node

```



1. 执行ts-node命令即可



```

ts-node hello.ts

```



### 第2集：详解TS中的类



**简介：详细讲解面向对象中类的概念**

```

既然要面向对象，操作对象，首先便要拥有对象，那么下一个问题就是如何创建一个对象。
要创建一个对象，必须先定义一个类，所谓的类可以理解为对象模型，程序中可以根据类创建指定类型的对象

```



1、如何定义类：

```

//通过 class 这个关键字来定义一个类
class Persion{
name:string;
age:number;

}

//通过这个类实例化一个对象
let mam=new Persion;
如果不传递参数的话（）可以省略

```



2、静态修饰符（static）



```

被 static 修饰的属性或者是方法，属于类的。可以通过类名调用，不属于实例的，实例没办使用

案例：
class Persion{
static name:string='ndedu';
static sayHello(){
console.log("嗨！！！！");
};
}

Persion.sayHello();

使用方法：放在属性名或者方法名前面

```



​	3、readonly



```

被 readonly 修饰的属性，只能读取不能修改

案例
person.name='张三'；

```





### 第4集：类中的构造方法

**简介：详细讲解ts中类中的构造方法**



1、如何在类中定义一个构造方法

```

1）:定义一个简单的构造方法
class Dog{
constructor(){
console.log("我创建一个 Dog");
}
}
const dog=new Dog();

//当我们调 new Dog();的时候我们就等于调用 Dog 中的构造方法
//在实例犯法中，this 就表示当前的实例
//在构造方法中当前对象及时当前新建的那个对象
//可以通过 this 向新建的对象中添加属性

2):定义一个有参构造方法

class Dog{
constructor(name:string){
console.log(name);
}
}
const dog=new Dog("大黄");
//创建的时候必须传递一个 string 类型的参数

```

2、this改造构造方法

```

class Dog{
name:string;
constructor(name:string){
this.name=name;
}
}

```



### 第5集：继承的简介

在TS中如何实现继承呢？

```

class Animal{
name:string
constructor(name:string){
this.name=name;
}
sleep(){
console.log(this.name+'在睡觉');
}
}
class Dog extends Animal{
constructor(name:string){
super(name);
}

}

//Dog extends Animal
此时，Animal 被称为父类，Dog 被称为子类
使用继承后，子类将会拥有父类所有的方法和属性
通过继承可以将多个类中共有的代码写在一个父类中
这样只需要写一次即可让所有子类都同时拥有父类中的属性和方法
如果希望在子类中添加一些父类中没又的属性或方法直接加就行

```





### 第6集：TS中的重写

**简介：详解ts中重写这个概念**



1、什么是重写？

```

子类覆盖了父类中的属性或者是方法叫做重写

```





### 第7集：super关键字

**简介：详细讲解TS中super关键字的使用**



1、super

```

1）：在当前类中 super 就表示当前类的父类

2）：如果在子类中写了构造方法，在子类构造方法中必须对父类的构造方法进行调用
子类不写构造方法，父类将自动调用，如果子类写构造方法，则会把父类构造方法覆盖调用，所有必须调用父类的构造方法
super();//有参数也需要传递对应的参数，不然会 TS 解析器会提示报错

```







### 第8集：抽象类

**简介：详细讲解TS中的抽象类的概念**



1、什么是抽象类

```

以 abstract 开头的类被称为抽象类
抽象类和其他类区别不大，只是不能用来创建对象
抽象类就是专门用来被继承的类

```



2、抽象方法

```

    抽象类中可以添加抽象方法
    抽象方法使用abstract开头，没有方法体
    抽象方法只能定义在抽象类中，子类必须对抽象方法进行重写
    abstract syaHello():void;

抽象类使用场景，当你不希望某个类被创建时，可以使用抽象类

```







### 第9集：TS中的接口精讲

**简介：详细讲解TS中的接口**



1、什么是接口？

```

接口用来定义一个类结构，用来定义一个类中应该包含哪些属性和方法。同时接口也可以当成类型声明去使用。接口可以重复声明，取所有相同接口名的并集

语法：
通过 interface 关键字来定义

案例：
interface Animal{

    }

interface myInterface{
name:string;
age:number;
sayHello():void;
}

//在定义类时，可以使类去实现一个接口
//实现接口就是使类满足接口的要求

    class pp implements myInterface{
        name:string;
        age:number;
        constructor(name,age){
            this.name=name;
            this.age=age;
        }
        sayHello(){
            console.log(this.name);
        }

    }
    const p=new pp("铁汁",56);
    console.log(p.sayHello);

```





### 第10集：快速掌握TS中属性的封装

**简介：详细讲解TS中的属性封装**



1、属性修饰符

```

public：修饰的属性可以在任意位置访问（修改），是默认值
private: 私有属性，私有属性只能在类内部进行访问（修改） -通过在类中添加方法使得私有属性可以被外部访问
protected: 受保护的属性，只能在当前类何当前类的之类中访问（修改）

```



2、属性封装

```

class Person {
private name:string;
private age:number;
constructor(name:string,age:number){
this.name=name;
this.age=age;
}
}

```





3、属性存取器

```

getter 方法用来读取属性

setter 方法用来设置属性 -它们被称为属性存取器

class Person {
private name:string;
private age:number;
constructor(name:string,age:number){
this.name=name;
this.age=age;
}
get name():string{
return this.name;
}

    set name(name:string):void{
        this.name=name;
    }

}

```







### 第11集：详解TS中的泛型

**简介：详解TS中泛型这个概念**



1、泛型使用场景

```

在定义函数或类时，如果遇到类型不明确就可以使用泛型
泛型就是一个不确定的类型

问题：引出
function cache(value:number):number{
return value;
}
// 需求 如果我实参可能是 boolean string object 这时该如何解决呢？
//解决方案： 指定 形参的类型和返回值类型为 any

//又出现的问题，如果使用 any，则会关闭 ts 的类型检测。不仅影响自己，还会在赋值时关闭其他变量的类型检测

//这个时候使用泛型即可解决我们以上的需求

```



2、如何使用简单的泛型

```

function cache<泛型名>(value:泛型名):泛型名{
return 泛型名;
}

```



3、泛型的约束

```

interface Person {
name: string,
age: number
}

function getPerson<T extends Person>(arg: T): T {
return arg
}
getPerson({
name: '张三',
age: 18
})

```



4、创建多个泛型

```

function getMessage<S, T>(name: S, age: T): [S, T] {
return [name, age]
}

console.log(getMessage<string, number>('张三', 18));

```



5、泛型在接口中的使用

```

interface Animal<T>{
name:T;
}

class Dog<T,F> implements Animal<T>{
name:T;
age:F;
constructor(name:T,age:F){
this.name=name;
this.age=age;
}
}

let jm=new Dog<string,number>("金毛",12);
console.log();

```







```

```