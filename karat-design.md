# Karat Design

## Load balancer

**Round Robin**

[Round-robin](https://kemptechnologies.com/load-balancing/round-robin-load-balancing/) load balancing is one of the simplest and most used load balancing algorithms. Client requests are distributed to application servers in rotation. For example, if you have three application servers: the first client request to the first application server in the list, the second client request to the second application server, the third client request to the third application server, the fourth to the first application server and so on.

**Weighted Round Robin**

Weighted Round Robin builds on the simple Round-robin load balancing algorithm to account for differing application server characteristics. The administrator assigns a weight to each application server based on criteria of their choosing to demonstrate the application servers traffic-handling capability.

**Least Connection**

Least Connection load balancing is a dynamic load balancing algorithm where client requests are distributed to the application server with the least number of active connections at the time the client request is received. 

## Consistent Hashing

 , **consistent hashing**[\[1\]](https://en.wikipedia.org/wiki/Consistent_hashing#cite_note-KargerEtAl1997-1)[\[2\]](https://en.wikipedia.org/wiki/Consistent_hashing#cite_note-nuggets-2) is a special kind of [hashing](https://en.wikipedia.org/wiki/Hash_function) such that when a [hash table](https://en.wikipedia.org/wiki/Hash_table) is resized, only {\displaystyle n/m}![n/m](https://wikimedia.org/api/rest_v1/media/math/render/svg/e555a7e118f9dbc0c67bc579d736ce73d94773e3) keys need to be remapped on average where {\displaystyle n}![n](https://wikimedia.org/api/rest_v1/media/math/render/svg/a601995d55609f2d9f5e233e36fbe9ea26011b3b) is the number of keys and {\displaystyle m}![m](https://wikimedia.org/api/rest_v1/media/math/render/svg/0a07d98bb302f3856cbabc47b2b9016692e3f7bc) is the number of slots. 

**Strong consistency and eventual consistency**

 **Strong Consistency** offers up-to-date data but at the cost of high latency. While **Eventual consistency** offers low latency but may reply to read requests with stale data since all nodes of the database may not have the updated data

