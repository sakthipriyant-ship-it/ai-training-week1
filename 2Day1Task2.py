class InsufficientFundsError(Exception):
    pass
class Account:
    def __init__(self,owner,balance=0):
        self.owner=owner
        self.balance=balance
    def deposit(self,amount):
        self.balance+=amount
        print(self.owner," deposited ",amount,"->balance ",self.balance)
    def withdraw(self,amount):
        if amount>self.balance:
            raise InsufficientFundsError(
                f"balance {self.balance}, requested {amount}"
            )
        self.balance-=amount
        print(self.owner,"withdraw ",amount," ->balance ",self.balance)
    def __str__(self):
        return f"Account Owner {self.owner} , balance {self.balance}"
class SavingsAccount(Account):
    def __init__(self,owner,balance=0):
        super().__init__(owner,balance)
        self.min_balance=500
    def withdraw(self, amount):
        if self.balance-amount<self.min_balance:
            raise InsufficientFundsError(
                f"balance {self.balance} , requested {amount}"
            )
        super().withdraw(amount)
priya=SavingsAccount("Priya",1500)
try:
    priya.withdraw(1200)
except InsufficientFundsError as e:
    print("Error : Insufficient funds : ",e)
try:
    priya.withdraw(800)
except InsufficientFundsError as e:
    print("Error : Insufficient funds : ",e)
print(isinstance(priya,Account))
print(priya)
