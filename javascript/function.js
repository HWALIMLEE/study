// Function
// - fundamental building block in the program
// - subprogram can be used multiple items (재사용 가능)
// - performs a task or calculates a value

//1. Function declaration 
// function name(param1, param2) {body...return;}
// one function === one thing
// naming : doSomething, command, verb
// function is object in JS

function printHello(){
    console.log("hello");
}
printHello();

function log(message){
    console.log(message);
}
log('Hello');
log(1234);


//2. Parameters
//premitive parameters : passed by value
//object parameters : passed by reference

function changeName(obj){
    obj.name = 'coder';
}

const ellie = {name : 'ellie'}; //ellie : refer, name: obj
changeName(ellie);
console.log(ellie);

//3. Default Parameters
function showMessage(message, from) {
    console.log(`${message} by ${from}`) 
}
showMessage('Hi!') // Hi! by unknown(고정값) 정해져 있지 않으면 고정값(undefined)으로 출력되게 된다. 

//4. Rest parameters (added in ES6)
function printAll(...args){ //...args - rest parameters, 배열 형태로 전달
    for (let i=0; i< args.length; i++){
        console.log(args[i])
    }
    for (const arg of args){
        console.log(arg);
    } // for문을 쓰기 보다 const of 이용해서 하나씩 출력 가능
    args.forEach((arg) => console.log(arg))
}
printAll('dream','coding','ellie') //배열 형태로 출력

// 5. Local scope
let globalMessage = 'global';
function printMessage(){
    let message = 'hello'; //local variable
    console.log(message);
    console.log(globalMessage);
    function printAnother(){
        console.log(message); // message는 확인할 수 있다. (자식은 부모에게서 정의된 메시지들을 확인 가능, 자식 안에 정의된 것은 부모에서 보려고 하면 오류가 발생한다. )
        let childMessage = 'hello';
    }
    // console.log(childMessage); //error!
}
printMessage(); // 밖에서는 안이 보이지 않고, 안에서만 밖을 볼 수 있다. 

//6. Return a value
function sum(a,b){
    return a+b;
}
const result = sum(1,2); //3

//7. Early return, early exit
//bad
function upgradeUser(user){
    if (user.point > 10){
    }
}
//good
function upgradeUser(user){
    if (user.point <=10 ){
        return;
    }
}

// First-class function
// functions are treated like any other variable
// can be assigned as a value to variable
// can be passed as an argument to other functions.
// can be returned by another function

// 1. Function expression
// a function declaration can be called earlier than it is defiend. (hoisted)
// a function expression is created when the execution reaches it.

// 함수에 이름이 없는 것 - 익명함수(anonymous function)
// function expression - hoisting가능, 함수 선언되기 이전에 명명 가능
const print = function(){
    console.log('print');
}
print();
const printAgain = print;
const sumAgain = sum;


// 2. Callback function using function expression
function randomQuiz(answer, printYes, printNo){
    if (answer === 'love you'){
        printYes();
    } else {
        printNo();
    }
}

//anonymous function
const printYes = function(){
    console.log('Yes');
};

// named function
// better debugging in debugger's stack traces
// recursions
const printNo = function print() {
    console.log('no!');
};

randomQuiz('wrong',printYes,printNo);
randomQuiz('love you',printYes, printNo);

// Arrow function
// always anonymous

// const simplePrint = function(){
//     console.log('simplePrint!');
// }
const simplePrint = () => console.log('simplePrint!')
const add = (a,b) => a+b;
const simpleMulitply = (a,b) => {
    return a * b;
};

// IIFE :
(function hello(){
    console.log("IIFE"); //선언함과 동시에 바로 호출, 바로바로 실행하고 싶을 때
})();
