from manim import *


def generate_coin_10_baht():
    coin_10_baht = Circle(radius=0.5, color=GOLD_A, fill_opacity=0.5)
    coin_10_baht_text = Text("10", font_size=48, color=WHITE)
    coin_10_baht_text.move_to(coin_10_baht.get_center())
    return coin_10_baht, coin_10_baht_text


def generate_coin_5_baht():
    coin_5_baht = Circle(radius=0.4, color=GRAY, fill_opacity=0.5)
    coin_5_baht_text = Text("5", font_size=40, color=WHITE)
    coin_5_baht_text.move_to(coin_5_baht.get_center())
    return coin_5_baht, coin_5_baht_text


def generate_coin_1_baht():
    bronze_color = "#CD7F32"
    coin_1_baht = Circle(radius=0.25, color=bronze_color, fill_opacity=0.5)
    coin_1_baht_text = Text("1", font_size=25, color=WHITE)
    coin_1_baht_text.move_to(coin_1_baht.get_center())
    return coin_1_baht, coin_1_baht_text


class NumberChangeCoinsGreedy(Scene):
    def construct(self):

        # title
        init_money_text = Text(
            "            วิธีการคนละโมบ (Greedy)\n\n          ในการทอนเงินจำนวน 28 บาท\n\nด้วยเหรียญ 1 บาท, 5 บาท, และ 10 บาท",
            font_size=48,
            color=WHITE,
        )
        init_money_text.move_to(ORIGIN)

        self.play(Write(init_money_text))
        self.wait(2)
        self.play(FadeOut(init_money_text))

        # start change algorithm
        first_money_text = Text("28 - 10 = 18", font_size=36)
        second_money_text = Text("18 - 10 = 8", font_size=36)
        third_money_text = Text("8 - 5 = 3", font_size=36)
        fourth_money_text = Text("3 - 1 = 2", font_size=36)
        fifth_money_text = Text("2 - 1 = 1", font_size=36)
        sixth_money_text = Text("1 - 1 = 0", font_size=36)

        # define the top position
        space_between_lines = DOWN * 1.25
        text_start_position = UP * 3 + LEFT * 3
        coin_start_position = text_start_position + RIGHT * 2.5

        # first
        coin_10_baht_1_1, coin_10_baht_text_1_1 = generate_coin_10_baht()
        self.play(
            FadeIn(
                first_money_text.move_to(text_start_position),
                coin_10_baht_1_1.move_to(coin_start_position),
                coin_10_baht_text_1_1.move_to(coin_start_position),
            )
        )
        self.wait(0.5)

        # second
        coin_10_baht_2_1, coin_10_baht_text_2_1 = generate_coin_10_baht()
        coin_10_baht_2_2, coin_10_baht_text_2_2 = generate_coin_10_baht()

        second_down_position = text_start_position + space_between_lines
        second_coin_start_position = second_down_position + RIGHT * 2.5
        self.play(
            FadeIn(
                second_money_text.move_to(second_down_position),
                coin_10_baht_2_1.move_to(second_coin_start_position),
                coin_10_baht_text_2_1.move_to(second_coin_start_position),
                coin_10_baht_2_2.move_to(second_coin_start_position + RIGHT * 1.5 * 1),
                coin_10_baht_text_2_2.move_to(
                    second_coin_start_position + RIGHT * 1.5 * 1
                ),
            )
        )
        self.wait(0.5)

        # third
        coin_10_baht_3_1, coin_10_baht_text_3_1 = generate_coin_10_baht()
        coin_10_baht_3_2, coin_10_baht_text_3_2 = generate_coin_10_baht()
        coin_5_baht_3_3, coin_5_baht_text_3_3 = generate_coin_5_baht()

        third_down_postion = text_start_position + space_between_lines * 2
        third_coin_start_position = third_down_postion + RIGHT * 2.5
        self.play(
            FadeIn(
                third_money_text.move_to(third_down_postion),
                coin_10_baht_3_1.move_to(third_coin_start_position),
                coin_10_baht_text_3_1.move_to(third_coin_start_position),
                coin_10_baht_3_2.move_to(third_coin_start_position + RIGHT * 1.5 * 1),
                coin_10_baht_text_3_2.move_to(
                    third_coin_start_position + RIGHT * 1.5 * 1
                ),
                coin_5_baht_3_3.move_to(third_coin_start_position + RIGHT * 1.5 * 2),
                coin_5_baht_text_3_3.move_to(
                    third_coin_start_position + RIGHT * 1.5 * 2
                ),
            )
        )
        self.wait(0.5)

        # fourth
        coin_10_baht_4_1, coin_10_baht_text_4_1 = generate_coin_10_baht()
        coin_10_baht_4_2, coin_10_baht_text_4_2 = generate_coin_10_baht()
        coin_5_baht_4_3, coin_5_baht_text_4_3 = generate_coin_5_baht()
        coin_1_baht_4_4, coin_1_baht_text_4_4 = generate_coin_1_baht()

        fourth_down_postion = text_start_position + space_between_lines * 3
        fourth_coin_start_position = fourth_down_postion + RIGHT * 2.5
        self.play(
            FadeIn(
                fourth_money_text.move_to(fourth_down_postion),
                coin_10_baht_4_1.move_to(fourth_coin_start_position),
                coin_10_baht_text_4_1.move_to(fourth_coin_start_position),
                coin_10_baht_4_2.move_to(fourth_coin_start_position + RIGHT * 1.5),
                coin_10_baht_text_4_2.move_to(fourth_coin_start_position + RIGHT * 1.5),
                coin_5_baht_4_3.move_to(fourth_coin_start_position + RIGHT * 1.5 * 2),
                coin_5_baht_text_4_3.move_to(
                    fourth_coin_start_position + RIGHT * 1.5 * 2
                ),
                coin_1_baht_4_4.move_to(
                    fourth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
                ),
                coin_1_baht_text_4_4.move_to(
                    fourth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
                ),
            )
        )
        self.wait(0.5)

        # fifth
        coin_10_baht_5_1, coin_10_baht_text_5_1 = generate_coin_10_baht()
        coin_10_baht_5_2, coin_10_baht_text_5_2 = generate_coin_10_baht()
        coin_5_baht_5_3, coin_5_baht_text_5_3 = generate_coin_5_baht()
        coin_1_baht_5_4, coin_1_baht_text_5_4 = generate_coin_1_baht()
        coin_1_baht_5_5, coin_1_baht_text_5_5 = generate_coin_1_baht()

        fifth_down_postion = text_start_position + space_between_lines * 4
        fifth_coin_start_position = fifth_down_postion + RIGHT * 2.5
        self.play(
            FadeIn(
                fifth_money_text.move_to(fifth_down_postion),
                coin_10_baht_5_1.move_to(fifth_coin_start_position),
                coin_10_baht_text_5_1.move_to(fifth_coin_start_position),
                coin_10_baht_5_2.move_to(fifth_coin_start_position + RIGHT * 1.5),
                coin_10_baht_text_5_2.move_to(fifth_coin_start_position + RIGHT * 1.5),
                coin_5_baht_5_3.move_to(fifth_coin_start_position + RIGHT * 1.5 * 2),
                coin_5_baht_text_5_3.move_to(
                    fifth_coin_start_position + RIGHT * 1.5 * 2
                ),
                coin_1_baht_5_4.move_to(
                    fifth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
                ),
                coin_1_baht_text_5_4.move_to(
                    fifth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
                ),
                coin_1_baht_5_5.move_to(
                    fifth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 1.75
                ),
                coin_1_baht_text_5_5.move_to(
                    fifth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 1.75
                ),
            )
        )
        self.wait(0.5)

        # sixth
        coin_10_baht_6_1, coin_10_baht_text_6_1 = generate_coin_10_baht()
        coin_10_baht_6_2, coin_10_baht_text_6_2 = generate_coin_10_baht()
        coin_5_baht_6_3, coin_5_baht_text_6_3 = generate_coin_5_baht()
        coin_1_baht_6_4, coin_1_baht_text_6_4 = generate_coin_1_baht()
        coin_1_baht_6_5, coin_1_baht_text_6_5 = generate_coin_1_baht()
        coin_1_baht_6_6, coin_1_baht_text_6_6 = generate_coin_1_baht()

        sixth_down_postion = text_start_position + space_between_lines * 5
        sixth_coin_start_position = sixth_down_postion + RIGHT * 2.5
        self.play(
            FadeIn(
                sixth_money_text.move_to(sixth_down_postion),
                coin_10_baht_6_1.move_to(sixth_coin_start_position),
                coin_10_baht_text_6_1.move_to(sixth_coin_start_position),
                coin_10_baht_6_2.move_to(sixth_coin_start_position + RIGHT * 1.5),
                coin_10_baht_text_6_2.move_to(sixth_coin_start_position + RIGHT * 1.5),
                coin_5_baht_6_3.move_to(sixth_coin_start_position + RIGHT * 1.5 * 2),
                coin_5_baht_text_6_3.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2
                ),
                coin_1_baht_6_4.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
                ),
                coin_1_baht_text_6_4.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
                ),
                coin_1_baht_6_5.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 1.75
                ),
                coin_1_baht_text_6_5.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 1.75
                ),
                coin_1_baht_6_6.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 2.5
                ),
                coin_1_baht_text_6_6.move_to(
                    sixth_coin_start_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 2.5
                ),
            )
        )

        self.wait(4)

        # fade out
        self.play(
            FadeOut(
                first_money_text,
                coin_10_baht_1_1,
                coin_10_baht_text_1_1,
            )
        )
        self.play(
            FadeOut(
                second_money_text,
                coin_10_baht_2_1,
                coin_10_baht_text_2_1,
                coin_10_baht_2_2,
                coin_10_baht_text_2_2,
            )
        )
        self.play(
            FadeOut(
                third_money_text,
                coin_10_baht_3_1,
                coin_10_baht_text_3_1,
                coin_10_baht_3_2,
                coin_10_baht_text_3_2,
                coin_5_baht_3_3,
                coin_5_baht_text_3_3,
            )
        )
        self.play(
            FadeOut(
                fourth_money_text,
                coin_10_baht_4_1,
                coin_10_baht_text_4_1,
                coin_10_baht_4_2,
                coin_10_baht_text_4_2,
                coin_5_baht_4_3,
                coin_5_baht_text_4_3,
                coin_1_baht_4_4,
                coin_1_baht_text_4_4,
            )
        )
        self.play(
            FadeOut(
                fifth_money_text,
                coin_10_baht_5_1,
                coin_10_baht_text_5_1,
                coin_10_baht_5_2,
                coin_10_baht_text_5_2,
                coin_5_baht_5_3,
                coin_5_baht_text_5_3,
                coin_1_baht_5_4,
                coin_1_baht_text_5_4,
                coin_1_baht_5_5,
                coin_1_baht_text_5_5,
            )
        )
        self.play(FadeOut(sixth_money_text))

        # end position
        sixth_coin_end_position = UP * 2 + LEFT * 3
        self.play(
            coin_10_baht_6_1.animate.move_to(sixth_coin_end_position),
            coin_10_baht_text_6_1.animate.move_to(sixth_coin_end_position),
            coin_10_baht_6_2.animate.move_to(sixth_coin_end_position + RIGHT * 1.5),
            coin_10_baht_text_6_2.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5
            ),
            coin_5_baht_6_3.animate.move_to(sixth_coin_end_position + RIGHT * 1.5 * 2),
            coin_5_baht_text_6_3.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2
            ),
            coin_1_baht_6_4.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
            ),
            coin_1_baht_text_6_4.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2 + RIGHT * 1.2
            ),
            coin_1_baht_6_5.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 1.75
            ),
            coin_1_baht_text_6_5.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 1.75
            ),
            coin_1_baht_6_6.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 2.5
            ),
            coin_1_baht_text_6_6.animate.move_to(
                sixth_coin_end_position + RIGHT * 1.5 * 2 + RIGHT * 1.2 * 2.5
            ),
        )

        # summary
        summary_money_text = Text(
            "สรุปจำนวนเหรียญที่น้อยที่สุด คือ 6 เหรียญ",
            font_size=48,
            color=WHITE,
        )
        summary_money_text.move_to(ORIGIN + DOWN)

        self.play(FadeIn(summary_money_text))
        self.wait(3)
        self.play(FadeOut(summary_money_text))
