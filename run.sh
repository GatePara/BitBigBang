startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

mkdir build
cd build 
cmake ..
make -j
#./mt_filter
./mt_filter6
# ./hnsw
endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s - $startTime_s ]

echo "$startTime ---> $endTime" "Total:$sumTime seconds"