class ABC:
    def add(self, x,y):
        return x+y
    def func(self):
        print(getattr(self, 'add')(2,3))

obj = ABC()
obj.func()