#!/usr/bin/env Rscript

library(ggplot2)
library(scales)

setwd("C:/Users/sergiym/devel/audioset_tagging_cnn/")

data <- read.csv("onnx_bench.csv")

data$model <- factor(data$model, levels=data$model)
data$type <- factor(data$type, levels=c(
  "DaiNet", "LeeNet", "MobileNet", "ResNet", "Res1dNet", "Cnn", "Wavegram"))

png("onnx_bench.png", height=1024, width=1600, res=220)

ggplot(data, aes(x=model, y=time/1000, fill=type)) +
  geom_bar(stat="identity", na.rm=T, alpha=0.9, color="black") +
  geom_text(label=round(data$time/1000, 1), nudge_y=12, alpha=0.9, size=2.3) +
  scale_x_discrete("Model") +
  scale_y_continuous("Inference time, ms") +
  scale_fill_brewer(palette="Set3") +
  labs(title="ONNX inference time per 10s frame", fill="Model Type") +
  theme_bw() +
  theme(legend.position="right", axis.text.x=element_text(angle=30, hjust=1))

dev.off()

png("onnx_size.png", height=1024, width=1600, res=220)

ggplot(data, aes(x=model, y=size/1024/1024, fill=type)) +
  geom_bar(stat="identity", na.rm=T, alpha=0.9, color="black") +
  geom_text(label=round(data$size/1024/1024, 1), nudge_y=10, alpha=0.9, size=2.3) +
  scale_x_discrete("Model") +
  scale_y_continuous("Model size, MB") +
  scale_fill_brewer(palette="Set3") +
  labs(title="ONNX model sizes", fill="Model Type") +
  theme_bw() +
  theme(legend.position="right", axis.text.x=element_text(angle=30, hjust=1))

dev.off()

png("onnx_mAP.png", height=1024, width=1600, res=220)

ggplot(data, aes(x=model, y=mAP, fill=type)) +
  geom_bar(stat="identity", na.rm=T, alpha=0.9, color="black") +
  geom_text(label=data$mAP, nudge_y=0.01, alpha=0.9, size=2.3) +
  scale_x_discrete("Model") +
  scale_y_continuous("mAP") +
  scale_fill_brewer(palette="Set3") +
  labs(title="Mean Average Precision", fill="Model Type") +
  theme_bw() +
  theme(legend.position="right", axis.text.x=element_text(angle=30, hjust=1))

dev.off()
