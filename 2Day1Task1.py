class InsufficientFundsError(Exception):
    pass
class Account:
    def __init__(self,owner,balance=0):
        self.owner=owner
        self.balance=balance
    def deposit(self,amount):
        self.balance+=amount
        print(self.owner," deposited ",amount,"->balance",self.balance)
    def withdraw(self,amount):
        if amount>self.balance:
            raise InsufficientFundsError(
                f"balance {self.balance}, requested {amount}"
            )
        self.balance-=amount
        print(self.owner,"Withdraw ",amount,"-> balance ",self.balance)

arun=Account("Arun",1000)

try:
    arun.deposit(500)
    arun.withdraw(2000)
except InsufficientFundsError as e:
    print("Error : Insufficient funds: ",e)
finally:
    print("Transaction finished")