put this in your local config
```sh
Host *-node*-p*
	IdentityFile ~/.ssh/YOUR_IDENTITY_FILE
	ConnectTimeout 60
	ServerAliveInterval 30
	ServerAliveCountMax 10
	HostbasedAuthentication yes
	ProxyCommand sh -c 'h="$1"; cluster="${h%%%%-*}"; rest="${h#*-}"; node="${rest%%%%-p*}"; port="${h##*-p}"; exec ssh -W "${node}:${port}" "$cluster"' -- %n
```