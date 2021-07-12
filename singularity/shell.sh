singularity instance stop -a;

if [ -d mysql-server ]; then
rm -r mysql-server;
fi
if [ -d apache-server ]; then
rm -r apache-server;
fi
if [ -d python-server ]; then
rm -r python-server;
fi

singularity build --sandbox mysql-server mysql-server.def;
singularity build --sandbox apache-server apache-server.def;
singularity build --sandbox python-server python-server.def;

singularity instance start --net --network-args "portmap=3306:3306/tcp" --network-args "IP=10.22.0.3" mysql-server/ mysql-server;
singularity instance start --net --network-args "portmap=80:80/tcp" --network-args "IP=10.22.0.2" apache-server/ apache-server;
singularity instance start --net --network-args "portmap=3000:3000/tcp" --network-args "IP=10.22.0.4" python-server/ python-server;