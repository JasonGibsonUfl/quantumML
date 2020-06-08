import pymysql as dba


class QueryEngine():
    # TODO: change to read only .cnf file
    db = dba.connect(read_default_file='/etc/mysql/my.cnf')
    cursor = db.cursor()
    #tables = cursor.fetchall()

    def get_calculation(self, band_gap_range=[]):
        # TODO: Throw error for more than 2 inputs
        query_str = 'SELECT * FROM calculations WHERE '
        if (len(band_gap_range) > 0):
            query_str += 'band_gap >= ' + str(band_gap_range[0])
            query_str += ' AND band_gap <= ' + str(band_gap_range[1])+';'

        self.cursor.execute(query_str)
        return self.cursor.fetchall()
