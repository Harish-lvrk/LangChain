from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name : str
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0,lt=10,default= 5,description="Decimal value is representation th cgpa of the student. ")

# it also provides the auto conversion when evver possible

new_student ={
    'name':"Hareesh",
    'age': "43",
    'email': "gamil@gmail.com",
    'cgpa' : 7
              }


Student1 = Student(**new_student)

print(Student1)


print(dict(Student1))


Student_json = Student1.model_dump_json()

