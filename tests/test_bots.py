import uuid


def test_bot_crud_scoped_to_user(client):
    email = f"user_{uuid.uuid4().hex[:8]}@example.com"

    r = client.post(
        "/auth/register",
        json={"name": "Test User", "email": email, "password": "password123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}

    r = client.post(
        "/bots",
        headers=headers,
        json={"name": "Bot A", "description": "desc"},
    )
    assert r.status_code == 200, r.text
    bot = r.json()
    bot_id = bot["id"]
    assert bot["owner_user_id"] is not None
    assert bot["api_key"]

    r = client.get("/bots", headers=headers)
    assert r.status_code == 200
    bots = r.json()
    assert any(b["id"] == bot_id for b in bots)

    r = client.get(f"/bots/{bot_id}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == bot_id

    r = client.delete(f"/bots/{bot_id}", headers=headers)
    assert r.status_code in (204, 200)

    r = client.get("/bots", headers=headers)
    assert r.status_code == 200
    bots = r.json()
    assert all(b["id"] != bot_id for b in bots)

