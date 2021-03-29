// 1. String Concatenation

console.log('my' + ' cat');
console.log('1' + 2);
console.log(`string literals : 1+2 = ${1+2}`)
console.log("ellie's notebook")


// 2. Numeric operators
console.log(1 + 1); // add
console.log(1 - 1); // substract
console.log(1 / 1); // divide
console.log(1 * 1); // multiply
console.log(5 % 2); // remainder
console.log(2 ** 3); // exponentiation

// 3. Increment and decrement operators
let counter=2;
const preIncrement = ++counter;
// counter = counter + 1;
// PreIncrement = counter;
console.log(`preIncrement: ${preIncrement}, counter: ${counter}`);

const postIncrement = counter++;
// postIncrement = counter;
// counter = counter + 1;
console.log(`postIncrement: ${postIncrement}, counter: ${counter}`);

const preDecrement = --counter;
console.log(`preDecrement: ${preDecrement}, counter: ${counter}`);

const postDecrement = counter--;
console.log(`postDecrement: ${postDecrement}, counter: ${counter}`);

// 4. Assignment operators
let x = 3;
let y = 6;

x+=y;
x-=y;
x*=y;
x/=y;

//5. Comparison operators
console.log(10 < 6); // less than
console.log(10 <= 6); // less than or equal
console.log(10 > 6); // greater than
console.log(10 >= 6); // greater than or equal

// 6. Logical Operators: || (or) , && (and), ! (not)
const value1 = false;
const value2 = 4 <2;

// || (or) , finds the first truty value
// expression이나 함수같은 복잡한 애들일 수록 뒤에 위치
console.log(`or: ${value1 || value2 || check()}`);

// &&(and), finds the first falsy value
// often used to compress long if-statment
// nullableObject && nullableOjbect.something
console.log(`and: ${value1 && value2 && check()}`);

function check(){
    for (let i=0 ; i<10; i++){
        console.log('😱');
    }
    return true;
}

// !(not) - 값을 반대로
console.log(!value1);

// 7. Equality
const stringFive = '5';
const numberFive = 5;

// == loose equality, with type conversion
console.log(stringFive == numberFive); //true
console.log(stringFive != numberFive); //false

console.log(stringFive === numberFive);
console.log(stringFive !== numberFive)

// object equality by reference
const ellie1 = { name: 'ellie' };
const ellie2 = { name: 'ellie' };
const ellie3 = ellie1;
console.log(ellie1 == ellie2); //false
console.log(ellie1 === ellie2); //false
console.log(ellie1 === ellie3); //true

// equality - puzzler
console.log(0 == false); //true
console.log(0 === false); //false
console.log('' == false); //true
console.log('' === false); //false
console.log(null == undefined); //true
console.log(null === undefined); //false

// 8. Conditional operators: if
// if, else if, else
const name = 'df';

if (name === 'ellie') {
    console.log("Welcome Ellie!");
} else if (name === 'coder') {
    console.log('You are amazing coder');
} else {
    console.log('unknwon');
}

// 9. Ternary operator: ?
// condition ? value1 : value2;
console.log(name === 'ellie' ? 'yes' : 'no');

// 10. Switch statement
// use for multiple if checks
// use for enum-like value check
// use for multiple type checks in TS

const browser = 'IE';
switch (browser) {
    case 'IE':
        console.log('go away');
        break;
    case 'Chrome':
        console.loe('love you!');
        break;
    case 'FireFox':
        console.log('love you!');
        break;
    default:
        console.log("same all!");
        break
    
}

switch (browser) {
    case 'IE':
        console.log('go away');
        break;
    case 'Chrome': //case 를 연달아 써줄 수도 있다
    case 'FireFox':
        console.loe('love you!');
        break;
    case 'FireFox':
        console.log('love you!');
        break;
    default:
        console.log("same all!");
        break
    
}


// 11. Loops
// while loop, while the condition is truty,
// body code is executed.

let i=3;
while (i>0){
    console.log(`while : ${i}`);
    i--;
}

// do while loop, body code is executed first, 
// then check the condition

do { // 블록 먼저 실행하고 조건이 맞는 지 체크, 블록 먼저 실행
    console.log(`do while: ${i}`)
    i--;
} while (i > 0);

// for loop, for(begin; condition; step)
for (i = 3; i > 0; i--) {
    console.log(`for: ${i}`);
  }

for (let i=3; i>0; i--){ //블록안에 지역변수 let 선언
    console.log(`for:${i}`)
}

// nested loops
for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
      console.log(`i: ${i}, j:${j}`);
    }
  }

// break, continue
// Q1. iterate from 0 to 10 and print only even numbers (use continue)
for (let i = 0; i < 11; i++) {
    if (i % 2 === 0) {
      continue;
    }
    console.log(`q1. ${i}`);
  }
  
  // Q2. iterate from 0 to 10 and print numbers until reaching 8 (use break)
  for (let i = 0; i < 11; i++) {
    if (i > 8) {
      break;
    }
    console.log(`q2. ${i}`);
  }
