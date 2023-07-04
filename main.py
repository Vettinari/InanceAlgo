from Archive.positions.continuous import ContinuousPositionUnbiased

# increase short
# increase long
# decrease short
# decrease long
# close short increase long
# close long increase short

if __name__ == '__main__':
    position = ContinuousPositionUnbiased(ticker='EURUSD', scaler=0.000001)
    cash_in_hand = 10000
    while True:
        action = round(float(input("Type action ")), 3)
        if position.validate_action(action=action, cash_in_hand=cash_in_hand):
            cash_in_hand -= position.modify_position(current_price=1.1, volume=action)
