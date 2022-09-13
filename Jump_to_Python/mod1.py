# 5장

def add(a, b):
    return a + b
def sub(a, b):
    return a - b

if __name__ == "__main__":
    print(add(1, 4))
    print(sub(4, 2))

# 프롬프트에서 "python mod1.py"로 직접 실행하면 "__name__" 변수에 "__main__" 값이 저장되어 조건문 참.
# 하지만 다른 파이썬 셸이나 다른 파이썬 모듈에서 mod1을 import할 경우 "__name__"변수에 mod1.py의 모듈 이름 값 "mod1" 이 저장 되어 조건문 거짓.