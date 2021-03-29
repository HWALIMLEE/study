'use strict';
// class : template
// object : instance of a class
// JavaScript classes
// - introduced in ES6
// - syntactical sugar over prototype-based inheritance


// 1. Class declarations
class Person{
    //constructor - 생성자를 이용하여 나중에 object만들 때 필요한 데이터 전달(name, age)
    constructor(name,age){
        //fields
        this.name = name; // 할당
        this.age = age; // 이름과 나이라는 fields가 있고, speak라는 method가 있는 것
    }
    //methods
    speak(){
        console.log(`${this.name}:hello!`) // this = 생성된 오브젝트(여기서는 ellie)
    }
}

const ellie = new Person('ellie',20); //새로운 오브젝트 만들 때 new를 사용
console.log(ellie.name);
console.log(ellie.age);
ellie.speak();


//2. Geter and Setters(방어적인 자세)
class User{
    constructor(firstName, lastName,age){
        this.firstName = firstName;
        this.lastName = lastName;
        this.age = age;
    }   
    get age() { // get을 통해 값을 return하고
        return this._age;
    }
    set age(value){  // set을 통해 값을 설정(값을 설정하기 떄문에 값을 받아와야 함)
        this._age = value <0 ? 0 : value;
    }
}

const user1 = new User('Steve','Job',-1);
console.log(user1.age); // 사람의 나이가 -1이 되는 것은 말이 안됨

// 3. Fields(public, private)
// Too soon!

// 5. Inheritance
// a way for one class to extend another class.
class Shape {
    constructor(width, height, color){ //세가지의 fields
        this.width = width;
        this.height = height;
        this.color = color;
    }
    draw() { //method
        console.log(`drawing ${this.color} color of`);
    }
    getArea(){ //method
        return this.width * this.height;
    }
}

class Rectangle extends Shape{} //Rectangle을 정의할 때 바로 shape를 상속받아서 귀찮음을 없앰
class Triangle extends Shape { //삼각형의 너비는 /2, 우리가 필요한 함수만 재정의해서 사용 가능(이것을 다형성이라고 부름)(overriding)
    draw(){
        super.draw(); // 부모의 메소드도 호출되고
        console.log('🔺') // 자식에서 새롭게 정의한 메소드도 출력
    }
    getArea(){
        return (this.width * this.height)/2;
    }
    toString(){
        return `Triangle:color:${this.color}`
    }
}

const rectangle = new Rectangle(20,20,'blue');
rectangle.draw();
console.log(rectangle.getArea());

const triangle = new Triangle(20,20,'red');
triangle.draw();
triangle.getArea()

//6. Class checking : instanceOf 오브젝트는 클래스를 이용해서 새로운 인스턴스
console.log(rectangle instanceof Rectangle); //왼쪽에 있는 오브젝트가 오른쪽에 있는 클래스의 인스턴스인지 아닌지
console.log(triangle instanceof Rectangle);
console.log(triangle instanceof Triangle);
console.log(triangle instanceof Shape); //true
console.log(triangle instanceof Object); //true  // 자바스크립트에서 만든 모든 오브젝트, 클래스들은 자바스크립트의 object를 상속한 것
console.log(triangle.toString());
