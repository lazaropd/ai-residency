import sys

l = float(sys.argv[1])
w = float(sys.argv[2])
h = float(sys.argv[3])

def volume(a,b,c):
    return a*b*c

if __name__ == "__main__":
   print(volume(l,w,h))
