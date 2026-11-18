import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyCropSuitability:

    def __init__(self):
        self.suhu = ctrl.Antecedent(np.arange(0, 51, 1), 'suhu')
        self.kelembapan = ctrl.Antecedent(np.arange(0, 101, 1), 'kelembapan')
        self.ph = ctrl.Antecedent(np.arange(0, 15, 0.1), 'ph')
        self.curah_hujan = ctrl.Antecedent(np.arange(0, 501, 1), 'curah_hujan')

        self.kesesuaian = ctrl.Consequent(np.arange(0, 101, 1), 'kesesuaian')

        self.suhu['rendah'] = fuzz.trapmf(self.suhu.universe, [0, 0, 15, 22])
        self.suhu['sedang'] = fuzz.trimf(self.suhu.universe, [18, 27, 35])
        self.suhu['tinggi'] = fuzz.trapmf(self.suhu.universe, [30, 38, 50, 50])

        self.kelembapan['rendah'] = fuzz.trapmf(self.kelembapan.universe, [0, 0, 30, 50])
        self.kelembapan['sedang'] = fuzz.trimf(self.kelembapan.universe, [35, 60, 80])
        self.kelembapan['tinggi'] = fuzz.trapmf(self.kelembapan.universe, [65, 85, 100, 100])

        self.ph['asam'] = fuzz.trapmf(self.ph.universe, [0, 0, 4.5, 6])
        self.ph['netral'] = fuzz.trimf(self.ph.universe, [5.5, 7, 8.5])
        self.ph['basa'] = fuzz.trapmf(self.ph.universe, [8, 9.5, 14, 14])

        self.curah_hujan['rendah'] = fuzz.trapmf(self.curah_hujan.universe, [0, 0, 50, 150])
        self.curah_hujan['sedang'] = fuzz.trimf(self.curah_hujan.universe, [100, 250, 350])
        self.curah_hujan['tinggi'] = fuzz.trapmf(self.curah_hujan.universe, [300, 400, 500, 500])

        self.kesesuaian['rendah'] = fuzz.trapmf(self.kesesuaian.universe, [0, 0, 25, 45])
        self.kesesuaian['sedang'] = fuzz.trimf(self.kesesuaian.universe, [35, 55, 75])
        self.kesesuaian['tinggi'] = fuzz.trapmf(self.kesesuaian.universe, [65, 85, 100, 100])

        rules = [
            ctrl.Rule(
                self.suhu['sedang'] &
                self.kelembapan['sedang'] &
                self.ph['netral'] &
                self.curah_hujan['sedang'],
                self.kesesuaian['tinggi']
            ),
            ctrl.Rule(
                self.suhu['rendah'] | self.suhu['tinggi'],
                self.kesesuaian['rendah']
            ),
            ctrl.Rule(
                self.kelembapan['rendah'] | self.kelembapan['tinggi'],
                self.kesesuaian['sedang']
            ),
            ctrl.Rule(
                self.ph['asam'] | self.ph['basa'],
                self.kesesuaian['rendah']
            ),
            ctrl.Rule(
                self.curah_hujan['rendah'] | self.curah_hujan['tinggi'],
                self.kesesuaian['sedang']
            ),
            ctrl.Rule(
                self.suhu['sedang'] &
                self.kelembapan['sedang'] &
                self.ph['netral'],
                self.kesesuaian['tinggi']
            ),
            ctrl.Rule(
                self.suhu['sedang'] &
                self.ph['netral'] &
                self.curah_hujan['sedang'],
                self.kesesuaian['tinggi']
            )
        ]

        self.system = ctrl.ControlSystem(rules)

    def predict(self, suhu, kelembapan, ph, curah_hujan):

        data = {
            'suhu': suhu,
            'kelembapan': kelembapan,
            'ph': ph,
            'curah_hujan': curah_hujan
        }

        batas = {
            'suhu': (0, 50),
            'kelembapan': (0, 100),
            'ph': (0, 14),
            'curah_hujan': (0, 500)
        }

        for nama, nilai in data.items():
            if not isinstance(nilai, (int, float)) or not np.isfinite(nilai):
                raise ValueError(f'{nama} harus berupa angka yang valid.')

            minimum, maksimum = batas[nama]
            if not minimum <= nilai <= maksimum:
                raise ValueError(
                    f'{nama} harus berada antara {minimum} dan {maksimum}.'
                )

        simulation = ctrl.ControlSystemSimulation(self.system)

        simulation.input['suhu'] = suhu
        simulation.input['kelembapan'] = kelembapan
        simulation.input['ph'] = ph
        simulation.input['curah_hujan'] = curah_hujan

        simulation.compute()

        score = simulation.output['kesesuaian']

        if score < 40:
            status = 'Rendah'
        elif score < 70:
            status = 'Sedang'
        else:
            status = 'Tinggi'

        return {
            'skor': round(float(score), 2),
            'status': status
        }


# Fungsi ini bisa dipanggil langsung dari Django
def hitung_kesesuaian(suhu, kelembapan, ph, curah_hujan):
    fuzzy = FuzzyCropSuitability()
    return fuzzy.predict(suhu, kelembapan, ph, curah_hujan)


if __name__ == '__main__':
    hasil = hitung_kesesuaian(27, 70, 6.5, 200)

    print('=== HASIL FUZZY LOGIC AGRISMART ===')
    print('Skor kesesuaian:', hasil['skor'])
    print('Tingkat kesesuaian:', hasil['status'])
