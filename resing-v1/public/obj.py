
class Num:
    @staticmethod
    def count(nums:list, bins:list):
        import numpy as np
        data = np.array(nums)
        counts, bin_edges = np.histogram(data, bins=bins)
        msg = ""
        for i in range(len(counts)):
            msg += (f"{bin_edges[i]} - {bin_edges[i+1]}: {counts[i]}\n")
        return msg, counts, bin_edges

class Nn:
    @staticmethod
    def _format_weights(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    @staticmethod
    def size_of_model(model, format_result = True):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return Nn._format_weights(pp) if format_result else pp
