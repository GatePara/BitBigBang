#pragma once
#include "hnswlib.h"

namespace hnswlib
{

    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++)
        {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX512)

    // Favor using AVX512 if available.
    static float
    L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)qty_ptr);
        float PORTABLE_ALIGN64 TmpRes[16];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m512 diff, v1, v2;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1)
        {
            v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            diff = _mm512_sub_ps(v1, v2);
            // sum = _mm512_fmadd_ps(diff, diff, sum);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
        }

        _mm512_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                    TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                    TmpRes[13] + TmpRes[14] + TmpRes[15];

        return (res);
    }
#endif

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1)
        {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

#endif

#if defined(USE_SSE)

    static float
    L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1)
        {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        size_t qty = *((size_t *)qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *)pVect1v + qty16;
        float *pVect2 = (float *)pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif

#if defined(USE_SSE)
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)qty_ptr);

        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1)
        {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        size_t qty = *((size_t *)qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *)pVect1v + qty4;
        float *pVect2 = (float *)pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float>
    {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        L2Space(size_t dim)
        {
            fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
            if (AVX512Capable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
            else if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#elif defined(USE_AVX)
            if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#endif

            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size()
        {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func()
        {
            return fstdistfunc_;
        }

        void *get_dist_func_param()
        {
            return &dim_;
        }

        ~L2Space() {}
    };

    static float
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr)
    {
        size_t qty = *((size_t *)qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *)pVect1;
        unsigned char *b = (unsigned char *)pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return float(res);
    }

    static float L2SqrI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr)
    {
        size_t qty = *((size_t *)qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *)pVect1;
        unsigned char *b = (unsigned char *)pVect2;

        for (size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return float(res);
    }

    //  static float L2SqrSIMD512SQ8(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    // {
    //     unsigned char *lhs = (unsigned char *)pVect1v;
    //     unsigned char *rhs = (unsigned char *)pVect2v;
    //     size_t dim = *((size_t *)qty_ptr);
    //     size_t qty32 = dim >> 5 << 5;
    //     __m512i sum = _mm512_setzero_epi32();
    //     for (size_t i = 0; i < qty32; i += 32)
    //     {
    //         __m256i A = _mm256_loadu_si256((__m256i *)(lhs + i));
    //         __m256i B = _mm256_loadu_si256((__m256i *)(rhs + i));
    //         __m512i AInt16 = _mm512_cvtepu8_epi16(A);
    //         __m512i BInt16 = _mm512_cvtepu8_epi16(B);
    //         __m512i difference = _mm512_sub_epi16(AInt16, BInt16);
    //         __m512i tmp = _mm512_madd_epi16(difference, difference);
    //         sum = _mm512_add_epi32(tmp, sum);
    //     }
    //     lhs += qty32;
    //     rhs += qty32;
    //     __m256i A = _mm256_set_epi8(*lhs, *(lhs + 1), *(lhs + 2), *(lhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    //     __m256i B = _mm256_set_epi8(*rhs, *(rhs + 1), *(rhs + 2), *(rhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    //     __m512i AInt16 = _mm512_cvtepu8_epi16(A);
    //     __m512i BInt16 = _mm512_cvtepu8_epi16(B);
    //     __m512i difference = _mm512_sub_epi16(AInt16, BInt16);
    //     __m512i tmp = _mm512_madd_epi16(difference, difference);
    //     sum = _mm512_add_epi32(tmp, sum);

    //     int ret = _mm512_reduce_add_epi32(sum);
    //     return float(ret);
    // }

    static float L2SqrSIMD512SQ8(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        unsigned char *lhs = (unsigned char *)pVect1v;
        unsigned char *rhs = (unsigned char *)pVect2v;
        size_t dim = *((size_t *)qty_ptr);
        size_t qty32 = dim >> 5 << 5;
        __m512i sum = _mm512_setzero_epi32();
        for (size_t i = 0; i < qty32; i += 32)
        {
            __m256i A = _mm256_loadu_si256((__m256i *)(lhs + i));
            __m256i B = _mm256_loadu_si256((__m256i *)(rhs + i));
            __m512i AInt16 = _mm512_cvtepu8_epi16(A);
            __m512i BInt16 = _mm512_cvtepu8_epi16(B);
            __m512i difference = _mm512_sub_epi16(AInt16, BInt16);
            sum = _mm512_dpwssd_epi32(sum,difference, difference);
        }
        lhs += qty32;
        rhs += qty32;
        __m256i A = _mm256_set_epi8(*lhs, *(lhs + 1), *(lhs + 2), *(lhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        __m256i B = _mm256_set_epi8(*rhs, *(rhs + 1), *(rhs + 2), *(rhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        __m512i AInt16 = _mm512_cvtepu8_epi16(A);
        __m512i BInt16 = _mm512_cvtepu8_epi16(B);
        __m512i difference = _mm512_sub_epi16(AInt16, BInt16);
        sum = _mm512_dpwssd_epi32(sum,difference, difference);

        int ret = _mm512_reduce_add_epi32(sum);
        return float(ret);
    }

    class L2SpaceI : public SpaceInterface<float>
    {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        L2SpaceI(size_t dim)
        {
            // if (dim % 4 == 0)
            // {
            //     fstdistfunc_ = L2SqrI4x;
            // }
            // else
            // {
            //     fstdistfunc_ = L2SqrI;
            // }
            fstdistfunc_ = L2SqrSIMD512SQ8;
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size()
        {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func()
        {
            return fstdistfunc_;
        }

        void *get_dist_func_param()
        {
            return &dim_;
        }

        ~L2SpaceI() {}
    };

#if defined(USE_AVX512)

    // static float L2SqrSIMD512SQ16(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    // {
    //     unsigned short *lhs = (unsigned short *)pVect1v;
    //     unsigned short *rhs = (unsigned short *)pVect2v;
    //     size_t dim = *((size_t *)qty_ptr);
    //     size_t qty16 = dim >> 4 << 4;
    //     __m512 sum;
    //     float unpack[16] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    //     sum = _mm512_load_ps(unpack);
    //     for (size_t i = 0; i < qty16; i += 16)
    //     {
    //         __m256i A = _mm256_loadu_si256((__m256i *)(lhs + i));
    //         __m256i B = _mm256_loadu_si256((__m256i *)(rhs + i));
    //         __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
    //         __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
    //         __m512 difference = _mm512_sub_ps(AFloat, BFloat);
    //         __m512 squaredDifference = _mm512_mul_ps(difference, difference);
    //         sum = _mm512_add_ps(sum, squaredDifference);
    //     }
    //     lhs += qty16;
    //     rhs += qty16;
    //     __m256i A = _mm256_setr_epi16(*lhs, *(lhs + 1), *(lhs + 2), *(lhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    //     __m256i B = _mm256_setr_epi16(*rhs, *(rhs + 1), *(rhs + 2), *(rhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    //     __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
    //     __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
    //     __m512 difference = _mm512_sub_ps(AFloat, BFloat);
    //     __m512 squaredDifference = _mm512_mul_ps(difference, difference);
    //     sum = _mm512_add_ps(sum, squaredDifference);

    //     _mm512_storeu_ps(unpack, sum);
    //     float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7] + unpack[8] + unpack[9] + unpack[10] + unpack[11] + unpack[12] + unpack[13] + unpack[14] + unpack[15];

    //     // lhs+=qty16;
    //     // rhs+=qty16;
    //     // for(size_t i=qty16;i<dim;i++)
    //     // {
    //     //     float t = float(*lhs) - float(*rhs);
    //     //     lhs++;
    //     //     rhs++;
    //     //     ret += t * t;
    //     // }
    //     return ret;
    // }
    // static float L2SqrSIMD512SQ16(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    // {
    //     unsigned short *lhs = (unsigned short *)pVect1v;
    //     unsigned short *rhs = (unsigned short *)pVect2v;
    //     size_t dim = *((size_t *)qty_ptr);
    //     size_t qty16 = dim >> 4 << 4;
    //     __m512 sum = _mm512_setzero();
    //     for (size_t i = 0; i < qty16; i += 16)
    //     {
    //         __m256i A = _mm256_loadu_si256((__m256i *)(lhs + i));
    //         __m256i B = _mm256_loadu_si256((__m256i *)(rhs + i));
    //         __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
    //         __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
    //         __m512 difference = _mm512_sub_ps(AFloat, BFloat);
    //         sum = _mm512_fmadd_ps(difference, difference, sum);
    //     }
    //     lhs += qty16;
    //     rhs += qty16;
    //     __m256i A = _mm256_setr_epi16(*lhs, *(lhs + 1), *(lhs + 2), *(lhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    //     __m256i B = _mm256_setr_epi16(*rhs, *(rhs + 1), *(rhs + 2), *(rhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    //     __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
    //     __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
    //     __m512 difference = _mm512_sub_ps(AFloat, BFloat);
    //     sum = _mm512_fmadd_ps(difference, difference, sum);

    //     float ret = _mm512_reduce_add_ps(sum);
    //     return ret;
    // }

    static float L2SqrSIMD512SQ16(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    {
        unsigned short *lhs = (unsigned short *)pVect1v;
        unsigned short *rhs = (unsigned short *)pVect2v;
        size_t dim = *((size_t *)qty_ptr);
        size_t qty16 = dim >> 4 << 4;
        __m512 sum = _mm512_setzero();
        for (size_t i = 0; i < qty16; i += 16)
        {
            __m256i A = _mm256_loadu_si256((__m256i *)(lhs + i));
            __m256i B = _mm256_loadu_si256((__m256i *)(rhs + i));
            __m512i AInt = _mm512_cvtepu16_epi32(A);
            __m512i BInt = _mm512_cvtepu16_epi32(B);
            __m512i differenceInt = _mm512_sub_epi32(AInt, BInt);
            __m512 difference = _mm512_cvtepi32_ps(differenceInt);
            sum = _mm512_fmadd_ps(difference, difference, sum);
        }
        lhs += qty16;
        rhs += qty16;
        __m256i A = _mm256_setr_epi16(*lhs, *(lhs + 1), *(lhs + 2), *(lhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        __m256i B = _mm256_setr_epi16(*rhs, *(rhs + 1), *(rhs + 2), *(rhs + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        __m512i AInt = _mm512_cvtepu16_epi32(A);
        __m512i BInt = _mm512_cvtepu16_epi32(B);
        __m512i differenceInt = _mm512_sub_epi32(AInt, BInt);
        __m512 difference = _mm512_cvtepi32_ps(differenceInt);
        sum = _mm512_fmadd_ps(difference, difference, sum);

        float ret = _mm512_reduce_add_ps(sum);
        return ret;
    }

    // inline static float L2SqrSIMD512SQ16(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
    // {

    //     size_t dim = *((size_t *)qty_ptr);
    //     __m512 sum = _mm512_setzero_ps();
    //     __m512 a_vec, b_vec;
    //     __m256i A, B;
    // compute_cycle:
    //     if (dim < 16)
    //     {
    //         __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, dim);
    //         A = _mm256_maskz_loadu_epi16(mask, pVect1v);
    //         B = _mm256_maskz_loadu_epi16(mask, pVect2v);
    //         a_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
    //         b_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
    //         dim = 0;
    //     }
    //     else
    //     {
    //         A = _mm256_loadu_epi16(pVect1v);
    //         B = _mm256_loadu_epi16(pVect2v);
    //         a_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
    //         b_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
    //         pVect1v += 32, pVect2v += 32, dim -= 16;
    //     }
    //     __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
    //     sum = _mm512_fmadd_ps(d_vec, d_vec, sum);
    //     if (dim)
    //         goto compute_cycle;

    //     return _mm512_reduce_add_ps(sum);
    // }

#endif
    class L2SpaceSQ16 : public SpaceInterface<float>
    {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        L2SpaceSQ16(size_t dim)
        {
            // if (dim % 4 == 0) {
            //     fstdistfunc_ = L2SqrI4x;
            // } else {
            //     fstdistfunc_ = L2SqrI;
            // }
            fstdistfunc_ = L2SqrSIMD512SQ16;
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned short);
        }

        size_t get_data_size()
        {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func()
        {
            return fstdistfunc_;
        }

        void *get_dist_func_param()
        {
            return &dim_;
        }

        ~L2SpaceSQ16() {}
    };

} // namespace hnswlib
