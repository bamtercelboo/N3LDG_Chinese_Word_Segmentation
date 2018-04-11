#!/bin/bash

basepath=$(cd `dirname $0`; pwd)
echo $basepath
nohup $basepath/../bin/NNSegmentor -l -train ../Data/CTB6_seg/train.ctb60.seg.hwc -dev ../Data/CTB6_seg/dev.ctb60.seg.hwc -test ../Data/CTB6_seg/test.ctb60.seg.hwc -model ../model/model -option ../option/option.save > log 2>&1 &
tail -f log


