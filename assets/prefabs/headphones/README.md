# Headphones Prefabs

Over-ear studio headphones and a 1/4 in audio cable that plugs into them, for use in SuperDex simulations. Each folder is a standalone prefab with its own CAD, collision and render models, tuning scenarios, readme and license:

- [`headphones/`](./headphones/README.md): the headphones, with a 1/4 in jack in the left cup.
- [`cable/`](./cable/README.md): a 1 m audio cable with 1/4 in TRS plugs at both ends, and a test block with a matching jack.

`headphones_set.superdex_prefab` sets them out on a table, nested, beside a wooden display head: the headphones lie on their back to the head's left (+Y) and the cable lies straight in front of it (+X). The origin is on the table at the head's base, and the head is fixed in place. The head is the headphones' `head_proxy`, turned teak under a gloss clear coat. The headphones can't start out worn on the head: a nested prefab can be placed but not posed, and their band rests shut, narrower than the head. Put on the head, their cups reach the table, so they stand on them with the band about 1 cm above the crown. `tuning/rest` leaves the set alone on the table: nothing moves more than a few millimeters as it settles.
