# app/main.py - ADDON
#
# Tambahkan import dan include router ini ke main.py
#

# Di bagian import, tambahkan:
from app.routes import query_routes

# Di bagian include_router, tambahkan:
app.include_router(query_routes.router)
