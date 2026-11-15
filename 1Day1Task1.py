sales=[12000,0,8500,15000,-200,9800]
a=10
b=10.5
c="Yoho"
d=[1,2,3]
e={
    "name":"Yoho",
    "Date":"20-10-26"
    }
f={1,2,3,4}
g=(1,"Yoho","20-10=26")
print(a,type(a))
print(b,type(b))
print(c,type(c))
print(d,type(d))
print(e,type(e))
print(f,type(f))
print(g,type(g))
print()
total=0
for i in sales:
    
    if i<=0:
        print("Skipped invalid value ",i)
        continue
    total+=i
    print("Added ",i," total ",total)
    if total>30000:
        break
if total>30000:
    print("Target reached. Stopped at ",total)
elif total==30000:
    print("Target reached")
else:
    print("Target not reached")
 