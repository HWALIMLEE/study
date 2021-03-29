//Objects
// one of the JavaScript's

// object 는 {key:value} 의 집합체 {property : property에 해당하는 값}
const name='ellie';
const age = 4;
print(name,age);
function print(name,age){
    console.log(name);
}

const ellie = {name:'ellie',age:4} //오브젝트로 관리 괄호를 이용해 바로 오브젝트 생성 가능

function print(person){
    console.log(person.name);
    console.log(person.age)
}
print(ellie);

//오브젝트 만드는 방법
const obj1 = {}; //'object literal' syntax 괄호 이용
const obj2 = new Object() //'object constrctor' syntax 클래스 템플릿 이용(object에서 정의한 constructor 호출)

//Runtime 에 타입 결정

ellie.hasJob = true; // 동적으로 추가 (가능하면 피해서 코딩할 것)
console.log(ellie.hasJob)

delete ellie.hasJob; // 동적으로 삭제도 가능
console.log(ellie.hasJob)


// 2. Computed properties
// key should be always string
console.log(ellie.name); 
console.log(ellie['name']); //string으로도 받아올 수 있다(computed properties)

ellie['hasJob'] = true;
console.log(ellie.hasJob);

function printValue(obj,key){
    console.log(obj.key); // obj property에 해당하는 키값이 들어있니..? -> no 이렇게 쓰기 보다는
    console.log(obj[key]) // 이렇게 쓰도록 하자
}
printValue(ellie,'name');


//3. Property value shorthand
const person1 = {name:'bob',age:2}


function makePeron(name,age){
    return {
        name: name, // key와 value의 이름이 동일하다면 삭제 가능
        age: age,
    };
}

const person4 = makePerson('ellie',30);
console.log(person4)

function makePerson(name,age){ // class 같은 것, class가 없을 때 이렇게 사용했음
    return{
        name,
        age,
    };
}

// Constructor Function
// class가 없을 때
function Person(name,age){
    // this = {}
    this.name = name;
    this.age = age;
    // return this
}

//호출할 때도
const person3 = new Person('ellie',20);
console.log(person3)

// 5. in operator: property existence check(key in obj)(해당 키가 오브젝트 안에 있는지)
console.log('name' in ellie);
console.log('age' in ellie);
console.log('random'in ellie);
console.log(ellie.random) // 정의하지 않은 키 


// 6. for..in vs for..of
// for (key in obj)
console.clear() //이전 것들 다 지우기
for (key in ellie){
    console.log(key);
}

// for(value of iterable) - 배열, 리스트
const array = [1,2,4,5];
for (let i=0; i< array.length; i++){ // 이전 코드
    console.log(array[i])
}

for (value of array){ // 훨씬 짧아짐
    console.log(value);
}

// Fun cloning(오브젝트 복사)
// Object.assgin(dst, [obj1, obj2, obj3...])
const user = {name:'ellie',age:'20'}
const user2 = user;
user2.name = 'coder';
console.log(user)

// old way
console.clear()
const user3 = {};
for (key in user){
    user3[key] = user[key];
}
console.log(user3);

// new way
const user4 = Object.assign({},user) //복사하고자 하는 target, 복사하려 하는 target
console.log(user4)

const fruit1 = {color :'red'}
const fruit2 = {color :'blue',size:'big'};
const mixed = Object.assign({},fruit1, fruit2); // 계속 덮어 씌우게 됨
console.log(mixed.color); 
console.log(mixed.size);
