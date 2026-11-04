# Kannibál Robotporszívó - 3D (Ursina) - fejlesztői jegyzet

## Indítás (VS Code terminál, Windows 11)
```
```
Első indításkor létrejön az `assets/` (textúrák, hangok), a `textures/ursina.ico` és a `settings.json` (hang BE/KI, hangerő).

## Fájlok
| Fájl | Szerep |
|---|---|
| `main.py` | belépési pont, Ursina világ, HUD, menü, bemenet, kamera |
| `agent.py` | robot-agent: döntéslogika, memória, üzenetfeldolgozás, szerepek hatása |
| `coordination.py` | szerepek (Felderítő/Takarító/Támogató/Vadász/Általános), profilok, koordinátor, párbeszéd-sablon |
| `environment.py` | szimulációs környezet, pályák (csempézett nagy világ), `CommunicationBus` |
| `evolution.py` | evolúciós algoritmus (változatlan) |
| `gui_manager.py` | reszponzív panelek, Tab / fül / panelenkénti láthatóság |
| `camera_rig.py` | belső/külső kamera tiszta logikája (állapotmentés, FPV póz, simítás) |
| `sound_manager.py` | generált hanghatások, cooldown, BE/KI, hangerő |
| `tests/` | `test_agents.py` (agentek, üzenetek), `test_controller_stub.py` (GUI/kamera/hang, csonkolt Ursinával), `sim_benchmark.py` |

## Billentyűk
Tab = GUI ki/be · F11 = teljes képernyő · O = GUI- és kamera-beállítások · K = kommunikációs elemek (chat + szerepcímkék) · B = hang BE/KI · W/A/S/D, nyilak = kamera · Q/E = elforgatás · görgő = zoom · jobb vagy középső gomb húzása = kamera mozgatása · V = kameramód · F = belső nézet · Z = követés · HOME = alaphelyzet · END = egész pálya · F3 = teljesítménymérő · Space = szünet · . = 1 tick · +/- = sebesség · R = újraindítás · N = következő generáció.
(Az `S` már nem léptet, a nyilak nem állítják a sebességet. A billentyűk a `main.py` elején lévő `KEYS` szótárban módosíthatók.)

## Multi-Agent rendszer
* **Szerepek** (`coordination.py`): a genomból, kvótákkal osztva ki (kb. 20% felderítő, 20% támogató, 15% vadász [csak CLASSIC/TDM], 35% takarító), 150 tickig zárolva; csak a koordinátor cserélheti, 60 tickenként, ha egy fontos szerep kiesett. A szerep a `ROLE_PROFILES` súlyain át módosítja a hasznosságokat (takarítás, segítség, vadászat, menekülés), a kommunikációs gyakoriságot és a felderítést.
* **Üzenetek** (`AgentMessage`, `CommunicationBus`): feladó, címzett/csoport, típus, strukturált mező (`pos`, `value`, `target_key`), küldési/kézbesítési/feldolgozási tick, lejárat, `processed_by`, `thread_id`. Hatótáv a KÜLDŐ helyétől, késleltetés és csomagvesztés, TDM-ben csapatszűrés.
* **Spam-védelem**: azonos üzenet ismétlése időablakon belül elnyelődik, tickenként max. 8 új alacsony prioritású üzenet; sürgős üzenetek (SOS, veszélyjelzés) saját időzítéssel mennek.
* **Kiesett robotok**: a címzett nélküli üzenet "elérhetetlen"-ként zárul; minden élő társ törli a kiesett robotra mutató foglalásokat, célokat, fenyegetéseket.

| Üzenet | Mi változik tényleg |
|---|---|
| `TASK_ASSIGN` / `TASK_OFFER` | a címzett energia, távolság, szerep, állapot alapján elfogad/elutasít; elfogadás után a döntése "Átvett célpont teljesítése" lesz |
| `TARGET_CLAIM` | mások memóriájában lefoglalt célpont; nem választják; közelebbi kérő elveszi a távolabbitól |
| `TARGET_RELEASE` | teljesítéskor / lejáratkor / energiahiánynál felszabadul |
| `ACK` (0) | a küldő megjegyzi az elutasítót, másik társat keres |
| `LOW_BAT_SOS` -> `OFFER_ENERGY` -> `HELP_ACCEPTED` / `ACK(HELP_COVERED)` | a segítő célt vált és odamegy (minden módban); a kérő lassít és közeledik; a felesleges második segítő visszavonul |
| `THREAT_DETECTED` | a gyengébb kitér, az erős (agresszív + energia) szembeszáll |
| `STATUS_UPDATE` | szerepváltás bejelentése |

## Világméret
A menüben "Pálya méret -/+" (1..5). A sablonpálya (88x60) n x n csempéből áll, váltakozva tükrözve, a csempék közti falakon ajtókkal; bázis bal oldalt, cél jobb oldalt, töltők minden csempében. A kosz és a körlimit a terület / skála arányában nő, a rádió hatótávja mérsékelten. A vektormezők számítása vektorizált.

## Kamera
Külső: állapot (pivot, elforgatás, dőlés, távolság) mentve FPV-re váltáskor és pontosan visszaállítva; FOLLOW/TOPDOWN nem írja felül. Belső: széles látószög (78°), enyhe lefelé dőlés, sebességkorlátos iránykövetés.
