import re

from setting_FIR import *
from setting_FIR import QueryFir
from tkinter import filedialog
import sqlite3
import re
import operator
from PIL import Image


class GetFileInfo:
    def __init__(self, master, file=''):
        self.master = master
        self.file = file
        self.db = BASEPATH + '\\' + 'OCR_MT.db'
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        self.qy = QueryFir()
        self.file_only = ''
        self.width = None
        self.height = None
        self.tipologia = 'NC'
        self.produttore = 'NOT FOUND'
        self.trasportatore = 'NOT FOUND'
        self.raccoglitore = 'NOT FOUND'

    def search_image(self):
        self.file = filedialog.askopenfilename(initialdir=IMAGEPATH, title="Seleziona FIR",
                                               filetypes=(("files png", "*.png"), ("files jpg", "*.jpg"),
                                                          ("files pdf", "*.pdf"), ('All files', '*.*')))
        self.file_only = self.file.split('/')[-1].split('.')[0]

    def word_like_cond(self, wlist, fieldname='parola', perc=False):
        wcond = {}
        for word in wlist:
            word_l = list(word)
            wlike = []
            or_lett = []
            for i, lett in enumerate(word_l):
                or_lett.append(lett)
                word_l[i] = '_'
                if i > 0:
                    word_l[i - 1] = or_lett[i - 1]
                wlike.append(''.join(word_l))
            # print(wlike)
            wcond[word] = {
                "{fieldname} like '{perc}{el}{perc}'".format(fieldname=fieldname,
                                                             el=el, perc='%' if perc else '') for el in wlike}
        return wcond

    def find_info(self):
        word_like = {}
        self.file_only = '_'.join(self.file.split('/')[-1].split('.')[0].split('_')[:2])

        # WINDOWS RICHIEDE SOSTITUZIONE BACKSLASH PER FILEPATH
        im = Image.open(self.file.replace('/', '\\'))
        self.width, self.height = im.size

        for tipo in TIPO_FIR:
            if tipo == 'NC':
                continue

            wlist = TIPO_FIR.get(tipo)['TEXT']
            wlist = wlist + TIPO_FIR.get(tipo)['NO_WORD']

            word_like[tipo] = self.word_like_cond(wlist)

            self.get_tipologia(tipo, word_like[tipo])
            self.aggiorna_tipologia_db()
            if self.tipologia == 'NC':
                print('TIPOLOGIA NON DETERMINATA --> NESSUNA INFO A DISPOSIZIONE')
                return
            self.get_info_fir()

    def esclusione_parole(self, tipo, word_like, pid, dlt_id):

        im = Image.open(self.file)
        self.width, self.height = im.size

        hw = int(self.width) / 2
        hh = int(self.height) / 2

        for nword in TIPO_FIR[tipo]['NO_WORD']:
            clike = '(' + ' or '.join(word_like[nword]) + ')'
            plike = """
                ( parola like '{s00}%{s01}' OR
                parola like '%{s10}' OR
                parola like '{s20}%')
            """.format(s00=nword[:3], s01=nword[-3:], s10=nword[-5:], s20=nword[5:])

            nowq = """
                {sub_body}
                WHERE
                ({clike} OR {plike} ) AND
                p.id < {pid} + {did} AND
                p.id > {pid} - {did} AND
                file = '{file}';
            """.format(sub_body=self.qy.sub_body, clike=clike, plike=plike, pid=pid, did=dlt_id, file=self.file_only)

            nwres = self.cur.execute(nowq).fetchall()
            if nwres:
                return nwres

        # PAROLA LONTANA DA QUELLE CERCATE (ES. "ROTTAMI") MA CHE,
        # SE TROVATA, ESCLUDE LA TIPOLOGIA (ES. "TIPOLOGIA A")
        nowq = """
            {sub_body}
            WHERE
            (parola = "{exc_word}") AND
            coor_x {sx} {hw} AND
            coor_y {sy} {hh} AND
            file = '{file}';
        """.format(sub_body=self.qy.sub_body, exc_word=TIPO_FIR[tipo]['NO_WORD'],
                   sx=TIPO_FIR[tipo]['SIGN'][0], hw=hw, sy=TIPO_FIR[tipo]['SIGN'][1], hh=hh, file=self.file_only)

        nwres = self.cur.execute(nowq).fetchall()
        if nwres:
            return nwres

        return None

    def get_tipologia(self, tipo, word_like):

        dlt_id = 25

        hw = int(self.width) / 2

        occ_l = []

        txt = TIPO_FIR[tipo]['TEXT'][0]
        clike = '(' + ' or '.join(word_like[txt]) + ')'
        plike = """
            ( parola like '{s00}%{s01}' OR
            parola like '{s00}%' OR
            parola like '%{s01}' )
        """.format(s00=txt[:3], s01=txt[-3:])

        subq = """
            {sub_body} WHERE
            ({clike} OR {plike}) AND
            coor_x {sx} {hw} AND
            file = '{file}'
            LIMIT 1;
        """.format(sub_body=self.qy.sub_body, clike=clike, plike=plike,
                   sx=TIPO_FIR[tipo]['SIGN'][0], hw=hw, file=self.file_only)

        sres = self.cur.execute(subq).fetchall()

        if not sres:
            print(
                "Il file {0} non appartiene ad nessuna tipologia --> TIPOLOGIA NC".format(self.file_only))
            return

        pid = sres[0][0]

        if TIPO_FIR[tipo]['NO_WORD']:
            nwres = self.esclusione_parole(tipo, word_like, pid, dlt_id)
            if nwres:
                print('La ricerca esclude il file {0} dalla tipologia : {1}'.format(self.file_only, tipo))
                return

        txt = TIPO_FIR[tipo]['TEXT'][1]
        clike = '(' + ' or '.join(word_like[txt]) + ')'
        plike = """
            (parola like '{s00}%{s01}' OR
            parola like '%{s10}%' OR
            parola like '{s20}%')
        """.format(s00=txt[:2], s01=txt[-2:], s10=txt[2:-2], s20=txt[:3])

        q = """
            {body}
            WHERE
            ({clike} OR {plike}) AND
            p.id < {pid} + {did} AND
            p.id > {pid} - {did} AND
            file = '{file}';
        """.format(body=self.qy.body, clike=clike, plike=plike, pid=pid, did=dlt_id, file=self.file_only)

        print(q)
        res = self.cur.execute(q).fetchall()

        occ_l.append(len(res))

        if occ_l == [0 * i for i in range(len(occ_l))]:
            print(
                "Il file {0} non appartiene ad nessuna tipologia --> TIPOLOGIA NC".format(self.file_only))
            return

        self.tipologia = tipo.split('_')[1]

    def aggiorna_tipologia_db(self):

        q = """
            UPDATE files_A3P
            SET tipologia = "{0}"
            WHERE file = "{1}"
        """.format(self.tipologia, self.file_only)

        self.cur.execute(q)
        self.conn.commit()

    def get_info_fir(self):

        prod = self.get_produttore()
        trasp = self.get_trasportatore()
        racc = self.get_raccoglitore()

        self.produttore = prod
        self.trasportatore = trasp
        self.raccoglitore = racc

    def get_produttore(self):

        bnd_word = {
            'INIZ': ['detentore'],
            'FIN': ['locale']
        }

        word_like = self.word_like_cond(bnd_word['INIZ'])

        wlike = ' or '.join([el for el in word_like.get(bnd_word['INIZ'][0])])

        subq = """
           {sub_body} WHERE
           file = '{file}' AND
           ({wlike})
           ORDER BY p.id DESC
           LIMIT 1;    
       """.format(sub_body=self.qy.sub_body, file=self.file_only, wlike=wlike)

        st_pid = self.cur.execute(subq).fetchall()[0][0]

        word_like = self.word_like_cond(bnd_word['FIN'])

        wlike = ' or '.join([el for el in word_like.get(bnd_word['FIN'][0])])

        subq = """
           {sub_body} WHERE
           file = '{file}' AND
           ({wlike})
           ORDER BY p.id DESC
           LIMIT 1;    
       """.format(sub_body=self.qy.sub_body, file=self.file_only, wlike=wlike)

        fi_pid = self.cur.execute(subq).fetchall()[0][0]

        q = """
            SELECT parola
            FROM parole_A3P p
            LEFT JOIN files_A3P f 
            ON (p.id_file=f.id)
            WHERE 
            file = '{file}' AND
            p.id BETWEEN {st_pid} + 1 AND {fi_pid} - 1;
        """.format(file=self.file_only, st_pid=st_pid, fi_pid=fi_pid)

        words = self.cur.execute(q).fetchall()
        print('TUTTE LE PAROLE', words)
        w_l = []
        for i, word in enumerate(words):
            if word[0] in INFO_FIR['PRODUTTORE']['NOWORD'] or re.match("\d", word[0]):
                continue
            elif len(word[0]) > 3:
                w_l.append(word[0])
            elif len(word[0]) == 3 and (re.match('l$', word[0]) or re.match('^s', word[0])):
                w_l.append(word[0])
            elif len(word[0]) == 1 and w_l:
                break

        print('PAROLE SELEZIONATE', w_l)

        # SE TROVO RISULTATO CON QUERY AVENTE TUTTE LE PAROLE (LIKE) NELLA STRINGA ALLORA SALTO SUBITO
        tlike = '(' + ' AND '.join(['rag_soc like "%{0}%"'.format(el) for el in w_l]) + ')'
        q = """
            SELECT rag_soc
            FROM PRODUTTORI
            WHERE
            {tlike}
            LIMIT 1
        """.format(tlike=tlike)

        res = self.cur.execute(q).fetchall()
        if res:
            self.produttore = res[0][0]
            print('{0} PRODUTTORE : {1} {0}'.format('-' * 20, self.produttore))
            return self.produttore

        word_like = self.word_like_cond(w_l, 'rag_soc', perc=True)
        tot_res = []
        k = 0
        plike = ''
        for txt in w_l:
            clike = '(' + ' or '.join(word_like[txt]) + ')'
            if len(txt) > 7 and k < 1:
                plike = 'OR ( rag_soc like "{s0}%" or rag_soc like "%{s1}")'.format(s0=txt[:-3], s1=txt[3:])
                k += 1

            q = """
                SELECT rag_soc
                FROM PRODUTTORI
                WHERE
                {clike} {plike}
            """.format(clike=clike, plike=plike if plike else '')

            res = self.cur.execute(q).fetchall()
            for w in res:
                tot_res.append(w[0])

        res_d = {}
        for el in tot_res:
            # PRENDO ELEMENTO PIU' FREQUENTE IN TUTTE LE RICERCHE
            res_d[el] = tot_res.count(el)

        print(res_d)
        prod_set = set()
        for key, val in res_d.items():
            if val == max(res_d.values()):
                prod_set.add(key)

        if len(prod_set) == 1:
            self.produttore = [prod for prod in prod_set][0]
        else:
            self.produttore = 'PREVISTI {} RISULTATI'.format(len(prod_set))

        print('{0} PRODUTTORE : {1} {0}'.format('-' * 20, self.produttore))

        return self.produttore

    def get_trasportatore(self):

        bnd_word = {
            'INIZ': ['trasportatore'],
            'FIN': ['indirizzo']
        }

        word_like = self.word_like_cond(bnd_word['INIZ'])

        wlike = ' or '.join([el for el in word_like.get(bnd_word['INIZ'][0])])

        subq = """
           {sub_body} WHERE
           file = '{file}' AND
           ({wlike})
           ORDER BY p.id DESC
           LIMIT 1;    
       """.format(sub_body=self.qy.sub_body, file=self.file_only, wlike=wlike)

        st_pid = self.cur.execute(subq).fetchall()[0][0]

        word_like = self.word_like_cond(bnd_word['FIN'])

        wlike = ' or '.join([el for el in word_like.get(bnd_word['FIN'][0])])

        subq = """
           {sub_body} WHERE
           file = '{file}' AND
           ({wlike})
           ORDER BY p.id DESC
           LIMIT 1;    
       """.format(sub_body=self.qy.sub_body, file=self.file_only, wlike=wlike)

        fi_pid = self.cur.execute(subq).fetchall()[0][0]

        q = """
            SELECT parola
            FROM parole_A3P p
            LEFT JOIN files_A3P f 
            ON (p.id_file=f.id)
            WHERE 
            file = '{file}' AND
            p.id BETWEEN {st_pid} + 1 AND {fi_pid} - 1;
        """.format(file=self.file_only, st_pid=st_pid, fi_pid=fi_pid)

        words = self.cur.execute(q).fetchall()
        print('TUTTE LE PAROLE', words)
        w_l = []
        # INFO FIR CERCATA PUO' TROVARSI IN MEZZO TRA LE RIGHE "PRODUTTORE" o "RAGIONE SOCIALE"
        for i, word in enumerate(words):
            if word[0] in INFO_FIR['TRASPORTATORE']['NOWORD'] or re.match("\d", word[0]):
                continue
            elif len(word[0]) > 3:
                w_l.append(word[0])
            elif len(word[0]) == 3 and (re.match('l$', word[0]) or re.match('^s', word[0])):
                w_l.append(word[0])
            elif len(word[0]) == 1 and w_l:
                break

        print('PAROLE SELEZIONATE', w_l)

        # SE TROVO RISULTATO CON QUERY AVENTE TUTTE LE PAROLE (LIKE) NELLA STRINGA ALLORA SALTO SUBITO
        tlike = '(' + ' AND '.join(['rag_soc like "%{0}%"'.format(el) for el in w_l]) + ')'

        q = """
            SELECT rag_soc
            FROM TRASPORTATORI
            WHERE
            {tlike}
            LIMIT 1
        """.format(tlike=tlike)

        res = self.cur.execute(q).fetchall()
        if res:
            self.trasportatore = res[0][0]
            print('{0} TRASPORTATORE : {1} {0}'.format('-' * 20, self.trasportatore))
            return self.trasportatore

        word_like = self.word_like_cond(w_l, 'rag_soc', perc=True)
        tot_res = []
        k = 0
        plike = ''
        for txt in w_l:
            clike = '(' + ' or '.join(word_like[txt]) + ')'
            if len(txt) > 7 and k < 1:
                plike = 'OR ( rag_soc like "{s0}%" or rag_soc like "%{s1}")'.format(s0=txt[:-3], s1=txt[3:])
                k += 1

            q = """
                SELECT rag_soc
                FROM TRASPORTATORI
                WHERE
                {clike} {plike}
            """.format(clike=clike, plike=plike if plike else '')

            res = self.cur.execute(q).fetchall()
            for w in res:
                tot_res.append(w[0])

        res_d = {}
        for el in tot_res:
            # PRENDO ELEMENTO PIU' FREQUENTE IN TUTTE LE RICERCHE
            res_d[el] = tot_res.count(el)

        print(res_d)
        trasp_set = set()
        for key, val in res_d.items():
            if val == max(res_d.values()):
                trasp_set.add(key)

        if len(trasp_set) == 1:
            self.trasportatore = [trasp for trasp in trasp_set][0]
        else:
            self.trasportatore = 'PREVISTI {} RISULTATI'.format(len(trasp_set))

        print('{0} TRASPORTATORE : {1} {0}'.format('-' * 20, self.trasportatore))

        return self.trasportatore

    def get_raccoglitore(self):

        bnd_word = {
            'INIZ': ['destinatario'],
            'FIN': ['destinazione']
        }

        word_like = self.word_like_cond(bnd_word['INIZ'])

        wlike = ' or '.join([el for el in word_like.get(bnd_word['INIZ'][0])])

        subq = """
                   {sub_body} WHERE
                   file = '{file}' AND
                   ({wlike})
                   ORDER BY p.id DESC
                   LIMIT 1;    
               """.format(sub_body=self.qy.sub_body, file=self.file_only, wlike=wlike)

        st_pid = self.cur.execute(subq).fetchall()[0][0]

        word_like = self.word_like_cond(bnd_word['FIN'])

        wlike = ' or '.join([el for el in word_like.get(bnd_word['FIN'][0])])

        subq = """
           {sub_body} WHERE
           file = '{file}' AND
           {wlike}
           ORDER BY p.id DESC
           LIMIT 1;    
       """.format(sub_body=self.qy.sub_body, file=self.file_only, wlike=wlike)

        fi_pid = self.cur.execute(subq).fetchall()[0][0]

        q = """
           SELECT parola
           FROM parole_A3P p
           LEFT JOIN files_A3P f 
           ON (p.id_file=f.id)
           WHERE 
           file = '{file}' AND
           p.id BETWEEN {st_pid} + 1 AND {fi_pid} - 1;
       """.format(file=self.file_only, st_pid=st_pid, fi_pid=fi_pid)

        words = self.cur.execute(q).fetchall()
        print('TUTTE LE PAROLE', words)
        w_l = []
        for i, word in enumerate(words):
            if word[0] in INFO_FIR['RACCOGLITORE']['NOWORD'] or re.match("\d", word[0]):
                continue
            elif len(word[0]) > 3:
                w_l.append(word[0])
            elif len(word[0]) == 3 and (re.match('l$', word[0]) or re.match('^s', word[0])):
                w_l.append(word[0])
            elif len(word[0]) == 1 and w_l:
                break

        print('PAROLE SELEZIONATE', w_l)

        # SE TROVO RISULTATO CON QUERY AVENTE TUTTE LE PAROLE (LIKE) NELLA STRINGA ALLORA SALTO SUBITO
        tlike = '(' + ' AND '.join(['rag_soc like "%{0}%"'.format(el) for el in w_l]) + ')'
        q = """
           SELECT rag_soc
           FROM RACCOGLITORI
           WHERE
           {tlike}
           LIMIT 1
       """.format(tlike=tlike)

        res = self.cur.execute(q).fetchall()
        if res:
            self.raccoglitore = res[0][0]
            print('{0} RACCOGLITORE : {1} {0}'.format('-' * 20, self.raccoglitore))
            return self.raccoglitore

        word_like = self.word_like_cond(w_l, 'rag_soc', perc=True)
        tot_res = []
        k = 0
        plike = ''
        for txt in w_l:
            clike = '(' + ' or '.join(word_like[txt]) + ')'
            if len(txt) > 7 and k < 1:
                plike = 'OR ( rag_soc like "{s0}%" or rag_soc like "%{s1}")'.format(s0=txt[:-3], s1=txt[3:])
                k += 1

            q = """
               SELECT rag_soc
               FROM RACCOGLITORI
               WHERE
               {clike} {plike}
           """.format(clike=clike, plike=plike if plike else '')

            res = self.cur.execute(q).fetchall()
            for w in res:
                tot_res.append(w[0])

        res_d = {}
        for el in tot_res:
            # PRENDO ELEMENTO PIU' FREQUENTE IN TUTTE LE RICERCHE
            res_d[el] = tot_res.count(el)

        print(res_d)
        racc_set = set()
        for key, val in res_d.items():
            if val == max(res_d.values()):
                racc_set.add(key)

        if len(racc_set) == 1:
            self.raccoglitore = [racc for racc in racc_set][0]
        else:
            self.raccoglitore = 'PREVISTI {} RISULTATI'.format(len(racc_set))

        print('{0} RACCOGLITORE : {1} {0}'.format('-' * 20, self.raccoglitore))

        return self.raccoglitore
