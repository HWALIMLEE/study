// 1. Use strict
// added in ES 5
// use this for Vanila Javascript.
'use strict;'

// 2. Var(read/write)
// let (added in ES6)
// block scope - 코드 블록 안에 작성하게 되면 코드 밖에서 볼 수 없음

let globalName = 'globaName'; // 항상 메모리에 탑재되어 있음, 많이 쓰지 않는 것이 좋음
{
    let name = 'hwalim';
    console.log(name);
    name = 'hello';
    console.log(name);
    console.log(globalName)
}

console.log(globalName)

// var hoisting - 항상 제일 위로 선언을 끌어올려주는 것(var 쓰면 안됨)
// var는 block scope 없음
// 변수 선언할 때는 let

// 3. Constant(Immutable data type)(read only)
// 값을 선언함과 동시에 할당한 이후로 바꿀 수 없음 -> 보안상의 이유로 아주 좋음
// 값이 변하지 않는 것을 사용하는 게 좋음

const daysInWeek = 7;
const MaxNumber = 5;

//Note!
// Immutable data types : premitive types(data자체는 변경 불가능), frozen objects (i.e. object.freeze())
// Mutable data types : all objects by default are mutable in JS (array 도 mutable)
// favor immutable data type alywas for a few reasons:
// - security
// - thread safety
// - reduce human mistakes



// 4. Variable Types
// primitive type : number, string, boolean, null, bigint(chrome, firefox만 지원)
// object - 한 박스로 관리(메모리에 한번에 올릴 수 없음)
// function, first-class function

//number 
const count = 17;
const size = 17.1;
console.log(`value: ${count}, type: ${typeof count}`);
console.log(`value: ${size}, type: ${typeof size}`);

// number - speicla numeric values: infinity, -infinity, NaN
const infinity = 1 / 0;
const negativeInfinity = -1 / 0;
const nAn = 'not a number' / 2;
console.log(infinity);
console.log(negativeInfinity);
console.log(nAn);

// string
const char = 'c';
const brendan = 'brendan';
const greeting = 'hello ' + brendan;
console.log(`value: ${greeting}, type: ${typeof greeting}`);
const helloBob = `hi ${brendan}!`; //template literals (string)
console.log(`value: ${helloBob}, type: ${typeof helloBob}`);
console.log('value: ' + helloBob + ' type: ' + typeof helloBob);

// boolean
// false: 0, null, undefined, NaN, ''
// true: any other value
const canRead = true;
const test = 3 < 1; // false
console.log(`value: ${canRead}, type: ${typeof canRead}`);
console.log(`value: ${test}, type: ${typeof test}`);

//null 
let nothing = null;
console.log(`value : ${nothing}, type:${typeof nothing}`) //type : object

// undefined
let x;
console.log(`value: ${x}, type: ${typeof x}`); //type : undefined

//string은 똑같지만 다른 symbol로 인식됨
// symbol, create unique identifiers for objects
const symbol1 = Symbol('id');
const symbol2 = Symbol('id');
console.log(symbol1 === symbol2);
const gSymbol1 = Symbol.for('id');
const gSymbol2 = Symbol.for('id');
console.log(gSymbol1 === gSymbol2); // true
console.log(`value: ${symbol1.description}, type: ${typeof symbol1}`); // id, symbol

// typescript - 자바스크립트 타입 위에 type이 올려진 언어

// object - 박스형태
// ellie가 가리키고 있는 포인터는 잠겨 있어서 다른 걸로 할당은 불가하지만 변수는 다른 값으로 할당이 가능
const ellie = {name : 'ellie',age:20}
ellie.age = 21; //변경 가능

// 5. Dynamic typing: dynamically typed language
let text = 'hello';
console.log(text.charAt(0)); //h
console.log(`value: ${text}, type: ${typeof text}`);
text = 1;
console.log(`value: ${text}, type: ${typeof text}`);
text = '7' + 5;
console.log(`value: ${text}, type: ${typeof text}`);
text = '8' / '2';
console.log(`value: ${text}, type: ${typeof text}`);
console.log(text.charAt(0));

