prices=[500,1200,800,2500,300]
def apply_discount(price, discount=10):
    a=price*discount/100
    print("apply_discount(1000) -> ",price-a)
apply_discount(1000)
apply_discount(1000,25)
def total(*args):
    print("total" ,args," -> ",sum(args))
total(1,4,5)
lt=[]
for i in prices:
    if i>1000:
        lt.append(i)
print("Above 1000: ",lt)
res=map(lambda prices:round(prices*118/100,2),prices)
print("With GST : ",list(res))
res2=filter(lambda prices:prices<1000,prices)
print("Below 1000 : ",list(res2))
