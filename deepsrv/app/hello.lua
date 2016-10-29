-- Get arguments
local args = ngx.req.get_uri_args()
ngx.log(ngx.ERR, "Hello World")

-- Get post body
ngx.req.read_body()
local data = ngx.req.get_body_data()

ngx.say(data)

