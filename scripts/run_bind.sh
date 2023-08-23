node=${1:-'SH-IDC1-10-198-34-39'}
port=${2:-'20488'}
echo http://cluster-proxy.sh.sensetime.com:$port
go-tcp-proxy_1.0.2_linux_amd64 -l 0.0.0.0:$port -r $node:$port