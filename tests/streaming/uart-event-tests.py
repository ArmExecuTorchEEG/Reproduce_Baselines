"""Exercise the actual streaming UART callback with a simulated CMSIS driver; no serial I/O."""
import pathlib, subprocess, tempfile

HEADERS = {'Driver_USART.h': '#include <stdint.h>\n#define ARM_DRIVER_OK 0\n#define ARM_POWER_FULL 2\n#define ARM_USART_ABORT_RECEIVE 4\n#define ARM_USART_ABORT_SEND 8\n#define ARM_USART_CONTROL_RX 16\n#define ARM_USART_CONTROL_TX 32\n#define ARM_USART_DATA_BITS_8 64\n#define ARM_USART_EVENT_RECEIVE_COMPLETE 128\n#define ARM_USART_EVENT_RX_BREAK 256\n#define ARM_USART_EVENT_RX_FRAMING_ERROR 512\n#define ARM_USART_EVENT_RX_OVERFLOW 1024\n#define ARM_USART_EVENT_RX_PARITY_ERROR 2048\n#define ARM_USART_EVENT_RX_TIMEOUT 4096\n#define ARM_USART_EVENT_SEND_COMPLETE 8192\n#define ARM_USART_FLOW_CONTROL_NONE 16384\n#define ARM_USART_MODE_ASYNCHRONOUS 32768\n#define ARM_USART_PARITY_NONE 65536\n#define ARM_USART_STOP_BITS_1 131072\ntypedef struct {int(*Initialize)(void(*)(uint32_t));int(*PowerControl)(int);int(*Control)(uint32_t,uint32_t);int(*Receive)(void*,uint32_t);int(*Send)(const void*,uint32_t);} ARM_DRIVER_USART;\n', 'RTE_Components.h': '#define CMSIS_device_header "device.h"\n', 'device.h': '#include <stdint.h>\nstatic uint32_t SystemCoreClock=1000000;static inline void SystemCoreClockUpdate(void){}static inline int SysTick_Config(uint32_t n){return n?0:1;}static inline void __DMB(void){}static inline uint32_t __get_PRIMASK(void){return 0;}static inline void __disable_irq(void){}static inline void __set_PRIMASK(uint32_t x){(void)x;}\n'}
TEST = '#include <assert.h>\n#include "firmware/streaming/uart_stream.c"\nstatic int init(void(*cb)(uint32_t)){(void)cb;return 0;}static int power(int x){(void)x;return 0;}static int ctl(uint32_t x,uint32_t y){(void)x;(void)y;return 0;}static int recv(void*p,uint32_t n){(void)p;(void)n;return 0;}static int send(const void*p,uint32_t n){(void)p;(void)n;return 0;}ARM_DRIVER_USART Driver_USART4={init,power,ctl,recv,send};int main(void){assert(stdio_init()==0);event(ARM_USART_EVENT_RX_TIMEOUT);assert(fault==0&&rx_idle_events==1);rx_byte=42;event(ARM_USART_EVENT_RECEIVE_COMPLETE);assert(stream_read_byte()==42);assert(stream_read_byte()==-1);event(ARM_USART_EVENT_RX_FRAMING_ERROR);assert(fault==1&&rx_error_events==1);assert(stream_uart_fault()==1);assert(fault==0);head=1023;tail=0;event(ARM_USART_EVENT_RECEIVE_COMPLETE);assert(rx_ring_overflows==1&&fault==1);return 0;}\n'

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="edgebci-uart-test-") as name:
        folder = pathlib.Path(name)
        for filename, content in HEADERS.items():
            (folder / filename).write_text(content)
        (folder / "test.c").write_text(TEST)
        subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-I" + str(folder),
                        "-I" + str(root), str(folder / "test.c"),
                        "-o", str(folder / "test")], check=True)
        subprocess.run([str(folder / "test")], check=True)
    print("PASS: actual UART callback retains receive on idle; transfers byte; invalidates framing errors and ring overflow. No physical port used.")

if __name__ == "__main__":
    main()
