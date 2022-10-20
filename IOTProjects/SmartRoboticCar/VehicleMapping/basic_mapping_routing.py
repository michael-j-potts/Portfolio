import picar_4wd as fc
from time import sleep

speed = 22

def main():
    while True:
        scan_list = fc.scan_step(32)
        if not scan_list:
            continue
    
        tmp = scan_list[0:10]
        print(tmp)
        if tmp[3:7] != [2,2,2,2]:
            print("backwards")
            fc.stop()
            sleep(0.3)
            fc.backward(speed)
            sleep(0.3)
            fc.stop()
            sleep(0.3)
            if tmp[1:3] != [2,2]:
                print("turn right")
                fc.turn_right(speed)
            else:
                print("turn left")
                fc.turn_left(speed)
                
        else:
            print("forwards")
            fc.forward(speed)
        

if __name__ == "__main__":
    try:
        main()
    finally:
        fc.stop()
