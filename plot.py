#! /usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

p_step_loss = re.compile(r"Training step (\d+): loss = (\d+\.\d+)")

def read_log(log_file, smooth = 5):
    epoches = []
    losses = []
    with open(log_file, "r") as f:
        for l in f:
            m = p_step_loss.search(l)
            if m:
                epoches.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                if epoches[-1] >= 20000 + 10*smooth//2:
                    break
    if smooth > 1:
        losses_smooth = np.convolve(losses, np.ones(smooth)/smooth, mode='same').tolist()
        losses_smooth[:smooth//2] = losses[:smooth//2]
        losses = losses_smooth[:-smooth//2]
        epoches = epoches[:-smooth//2]
    return epoches, losses

def plot1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for i, (log_file, label) in enumerate([("log-negff.txt", "NegGPT2"), ("log-original.txt", "GPT2"), ("log-negattn.txt", "NegAttn"), ("log-negattnff.txt", "NegAttnFF")]):
        epoches, losses = read_log(log_file)
        alpha = 1 - 0.1 * i
        print(label, alpha)
        ax1.plot(epoches, losses, label=label, alpha=alpha)
        ax2.plot(epoches, losses, alpha=alpha)

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # 使用 FuncFormatter 来自定义 y 轴刻度格式
    formatter = ticker.FuncFormatter(lambda y, _: '%d'%y)
    ax1.yaxis.set_minor_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    formatter = ticker.FuncFormatter(lambda y, _: f'{y:.2f}')
    ax2.yaxis.set_minor_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)

    ax1.legend()
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test loss")
    # ax1.set_xlim(0,20000)
    ax2.set_xlim(0,2000)
    # plt.savefig("gpt2.png", dpi=300)
    plt.savefig("gpt2.pdf")


if __name__ == "__main__":
    plot1()
