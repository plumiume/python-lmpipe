from multiprocessing import Process

class Test:
    def __new__(cls, obj: object):
        # __new__にパラメータがあるときの
        # このクラスの初期バージョン作成に
        # このメソッドがエラーとなるか検証
        return super().__new__(cls)

def process_target(test: Test):
    print("In child process:", test)

if __name__ == "__main__":
    test = Test("test object")
    process = Process(target=process_target, args=(test,))
    process.start()
    process.join()
    
