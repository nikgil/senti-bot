import os
import discord
import sys

from discord import Game
from dotenv import load_dotenv
from os.path import join, dirname
from discord.ext.commands import Bot

# This is from rolley
from kernels.linear_svc import LinearSVC

PREFIX = '?'

bot = Bot(command_prefix=PREFIX)

kernel: LinearSVC = LinearSVC()


# Bot Events
@bot.event
async def on_ready():
    await bot.change_presence(game=Game(name="Kaggle"))
    print("Logged in as")
    print(bot.user.name)
    print(bot.user.id)
    print("------")


@bot.event
async def on_message(message):
    if message.author.bot:
        # Ignore bots
        return

    if message.channel.is_private:
        # TODO: private channel is where training is
        return

    # TODO: channel excpetions & user exceptions
    # TODO: filter message here using algorithm
    await bot.process_commands(message)

    if kernel.is_banned(message.content):
        await bot.delete_message(message)


# Bot Commands
@bot.command(name='test_msg', description='Test a message to see if it is legal in chat', aliases=['test'],
             brief='test a message',
             pass_context=True)
async def test_msg(ctx):
    if len(ctx.message.content) == 0:
        await bot.send_message(ctx.message.channel, content='Please supply an actual message...')

    msg = ctx.message.content + "\n\n"
    color = 0x00FF00

    if kernel.is_banned(msg):
        color = 0xFF0000
        msg += "Is banned in this channel"
    else:
        msg += "Is okay to use in this channel"

    await bot.send_message(ctx.message.channel, embed=get_embed(msg, color))


def get_embed(content, color=0xffd700):
    return discord.Embed(title='**Sentiment Bot**', type='rich',
                         description=content, color=color)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TOKEN = sys.argv[1]
    else:
        dotenv_path = join(dirname(__file__), '.env')
        load_dotenv(dotenv_path)

        TOKEN = os.environ.get('TOKEN')

    bot.run(TOKEN)
