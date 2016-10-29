killall -9 nginx

rm -rf install
mkdir install
mkdir -p install/conf
mkdir -p install/logs
ln -s ../../nginx.conf install/conf/
ln -s ../web install/
ln -s ../app install/

nginx -p install
